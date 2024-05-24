from typing import List, Union, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
import json
import torch
from jsonformer import Jsonformer
from jsonformer.logits_processors import (
    NumberStoppingCriteria,
    OutputNumbersTokens,
    StringStoppingCriteria,
)

class BatchedJsonformer(Jsonformer):
    """
    Batches several prompts across the same json schema
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        json_schema: Dict[str, Any],
        prompts: List[str],
        *,
        debug: bool = False,
        max_array_length: int = 10,
        max_number_tokens: int = 6,
        temperature: float = 1.0,
        max_string_token_length: int = 10,
    ):
        super().__init__(
            model,
            tokenizer,
            json_schema,
            prompts[0],  # Pass the first prompt to the parent class
            debug=debug,
            max_array_length=max_array_length,
            max_number_tokens=max_number_tokens,
            temperature=temperature,
            max_string_token_length=max_string_token_length,
        )
        self.prompts = prompts

    def get_prompt(self, index: int):
        template = """{prompt}\nOutput result in the following JSON schema format:\n{schema}\nResult: {progress}"""
        progress = json.dumps(self.value[index])
        gen_marker_index = progress.find(f'"{self.generation_marker}"')
        if gen_marker_index != -1:
            progress = progress[:gen_marker_index]
        else:
            raise ValueError("Failed to find generation marker")

        prompt = template.format(
            prompt=self.prompts[index],
            schema=json.dumps(self.json_schema),
            progress=progress,
        )

        return prompt

    def generate_number(self, temperature: Union[float, None] = None, iterations=0):
        input_tokens = [self.tokenizer.encode(self.get_prompt(i), return_tensors="pt").to(self.model.device) for i in range(len(self.prompts))]
        input_tokens = torch.cat(input_tokens, dim=0)
        self.debug("[generate_number]", input_tokens, is_prompt=True)
        
        response = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_number_tokens,
            num_return_sequences=1,
            logits_processor=[self.number_logit_processor],
            stopping_criteria=[
                NumberStoppingCriteria(self.tokenizer, len(input_tokens[0])) # TODO
            ],
            temperature=temperature or self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        responses = [self.tokenizer.decode(resp, skip_special_tokens=True).strip().rstrip(".") for resp in response]
        self.debug("[generate_number]", responses)
        
        results = []
        for resp in responses:
            try:
                results.append(float(resp))
            except ValueError:
                if iterations > 3:
                    raise ValueError("Failed to generate a valid number")
                results.append(self.generate_number(temperature=self.temperature * 1.3))
        
        return results

    def generate_boolean(self) -> List[bool]:
        input_tokens = [self.tokenizer.encode(self.get_prompt(i), return_tensors="pt").to(self.model.device) for i in range(len(self.prompts))]
        input_tokens = torch.cat(input_tokens, dim=0)
        self.debug("[generate_boolean]", input_tokens, is_prompt=True)

        output = self.model.forward(input_tokens)
        logits = output.logits[:, -1, :]

        true_token_id = self.tokenizer.convert_tokens_to_ids("true")
        false_token_id = self.tokenizer.convert_tokens_to_ids("false")

        results = (logits[:, true_token_id] > logits[:, false_token_id]).tolist()
        self.debug("[generate_boolean]", results)

        return results

    def generate_string(self) -> List[str]:
        input_tokens = [self.tokenizer.encode(self.get_prompt(i) + '"', return_tensors="pt").to(self.model.device) for i in range(len(self.prompts))]
        input_tokens = torch.cat(input_tokens, dim=0)
        self.debug("[generate_string]", input_tokens, is_prompt=True)

        response = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_string_token_length,
            num_return_sequences=1,
            temperature=self.temperature,
            stopping_criteria=[
                StringStoppingCriteria(self.tokenizer, len(input_tokens[0]))
            ],
            pad_token_id=self.tokenizer.eos_token_id,
        )

        responses = []
        for i, resp in enumerate(response):
            if (
                len(resp) >= len(input_tokens[i])
                and (resp[: len(input_tokens[i])] == input_tokens[i]).all()
            ):
                resp = resp[len(input_tokens[i]):]
            if resp.shape[0] == 1:
                resp = resp[0]

            decoded_resp = self.tokenizer.decode(resp, skip_special_tokens=True)
            self.debug("[generate_string]", "|" + decoded_resp + "|")

            if decoded_resp.count('"') < 1:
                responses.append(decoded_resp)
            else:
                responses.append(decoded_resp.split('"')[0].strip())

        return responses

    def generate_array(self, item_schema: Dict[str, Any], obj_list: List[List[Any]]) -> List[List[Any]]:
        for i in range(len(self.prompts)):
            obj_list[i] = []

        for _ in range(self.max_array_length):
            elements = self.generate_value(item_schema, obj_list)
            for i in range(len(self.prompts)):
                obj_list[i].append(elements[i])

            input_prompts = [self.get_prompt(i) for i in range(len(self.prompts))]
            input_tokens = [self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device) for prompt in input_prompts]
            input_tokens = torch.cat(input_tokens, dim=0)

            output = self.model.forward(input_tokens)
            logits = output.logits[:, -1, :]

            top_indices = logits.topk(30).indices
            sorted_token_ids = [top_indices[i][logits[i][top_indices[i]].argsort(descending=True)] for i in range(len(self.prompts))]

            found_comma = [False] * len(self.prompts)
            found_close_bracket = [False] * len(self.prompts)

            for i in range(len(self.prompts)):
                for token_id in sorted_token_ids[i]:
                    decoded_token = self.tokenizer.decode(token_id)
                    if ',' in decoded_token:
                        found_comma[i] = True
                        break
                    if ']' in decoded_token:
                        found_close_bracket[i] = True
                        break

            if all(found_close_bracket) or not any(found_comma):
                break

        return obj_list
    
    def generate_object(
        self, properties: Dict[str, Any], obj_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        for key, schema in properties.items():
            self.debug("[generate_object] generating value for", key)
            # obj[key] = self.generate_value(schema, obj, key)
            values = self.generate_value(schema, obj_list, key) # TODO
            for i, value in enumerate(values):
                obj_list[i][key] = value
        return obj_list

    def generate_value(
        self,
        schema: Dict[str, Any],
        obj_list: Union[List[Dict[str, Any]], List[Any]],
        key: Union[str, None] = None,
    ) -> Any:
        schema_type = schema["type"]
        if schema_type == "number":
            if key:
                # obj_list[i][key] = self.generation_marker for i in range(len(obj_list))
                for i in range(len(obj_list)):
                    obj_list[i][key] = self.generation_marker
            else:
                # obj.append(self.generation_marker)
                for i in range(len(obj_list)):
                    obj_list[i].append(self.generation_marker)
            return self.generate_number()
        elif schema_type == "boolean":
            if key:
                # obj[key] = self.generation_marker
                for i in range(len(obj_list)):
                    obj_list[i][key] = self.generation_marker
            else:
                # obj.append(self.generation_marker)
                for i in range(len(obj_list)):
                    obj_list[i].append(self.generation_marker)
            return self.generate_boolean()
        elif schema_type == "string":
            if key:
                # obj[key] = self.generation_marker
                for i in range(len(obj_list)):
                    obj_list[i][key] = self.generation_marker
            else:
                # obj.append(self.generation_marker)
                for i in range(len(obj_list)):
                    obj_list[i].append(self.generation_marker)
            return self.generate_string()
        elif schema_type == "array":
            new_arrays = [[]*len(obj_list)]
            # obj[key] = new_array
            for i in range(len(obj_list)):
                obj_list[i][key] = new_arrays[i]
            return self.generate_array(schema["items"], new_arrays)
        elif schema_type == "object":
            new_objs = [{} for _ in range(len(obj_list))]
            if key:
                # obj[key] = new_obj
                for i in range(len(obj_list)):
                    obj_list[i][key] = new_objs[i]
            else:
                # obj.append(new_obj)
                for i in range(len(obj_list)):
                    obj_list[i].append(new_objs[i])
            return self.generate_object(schema["properties"], new_objs)
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    def __call__(self) -> List[Dict[str, Any]]:
        self.value = [{} for _ in range(len(self.prompts))]
        generated_data = self.generate_object(self.json_schema["properties"], self.value)
        return generated_data
               