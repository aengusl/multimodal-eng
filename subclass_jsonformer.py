from typing import List, Union, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
import json
import torch
from jsonformer import Jsonformer
from jsonformer.logits_processors import (
    NumberStoppingCriteria,
    OutputNumbersTokens,
    StringStoppingCriteria,
)

class CogVLMJsonformer(Jsonformer):
    """
    CogVLMJsonformer is a subclass of Jsonformer that uses CogVLM models to generate JSON data.
    """
    def __init__(self, images: Optional[List["PIL.Image"]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if images is not None:
            assert len(images) == 1, "VLMJsonformer only supports a single image for now..."
        self.image = images[0] if images is not None else None
    

    def generate_number(self, temperature: Union[float, None] = None, iterations=0):
        prompt = self.get_prompt()
        self.debug("[generate_number]", prompt, is_prompt=True)
        build_input_ids = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=prompt,
            images=[self.image] if self.image is not None else None,
            template_version="base",
        )
        input_tokens = build_input_ids['input_ids'].to(self.model.device)
        inputs = {
            'input_ids': build_input_ids['input_ids'].unsqueeze(0).to(self.model.device),
            'token_type_ids': build_input_ids['token_type_ids'].unsqueeze(0).to(self.model.device),
            'attention_mask': build_input_ids['attention_mask'].unsqueeze(0).to(self.model.device),
            'images': [[build_input_ids['images'][0].to(self.model.device).to(torch.bfloat16)]] if self.image is not None else None,
        }
        gen_kwargs = {
            "max_new_tokens": self.max_number_tokens,
            "temperature": temperature or self.temperature,
            "pad_token_id": self.tokenizer.eos_token_id,
            "logits_processor": [self.number_logit_processor],
            "do_sample": True,
            "num_return_sequences": 1,
            "stopping_criteria": [
                StringStoppingCriteria(self.tokenizer, len(input_tokens))
            ],
        }
        response = self.model.generate(**inputs, **gen_kwargs)
        response = self.tokenizer.decode(response[0], skip_special_tokens=True)
        torch.cuda.empty_cache()

        response = response[len(prompt) :]
        response = response.strip().rstrip(".")
        self.debug("[generate_number]", response)
        try:
            return float(response)
        except ValueError:
            if iterations > 3:
                # raise ValueError("Failed to generate a valid number")
                return 123.0 # TODO: hardcode
            iterations += 1

            return self.generate_number(temperature=self.temperature * 1.3, iterations=iterations)
        
    def generate_string(self) -> str:
        prompt = self.get_prompt() + '"'
        self.debug("[generate_string]", prompt, is_prompt=True)
        build_input_ids = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=prompt,
            images=[self.image] if self.image is not None else None,
            template_version="base",
        )
        input_tokens = build_input_ids['input_ids'].to(self.model.device)
        inputs = {
            'input_ids': build_input_ids['input_ids'].unsqueeze(0).to(self.model.device),
            'token_type_ids': build_input_ids['token_type_ids'].unsqueeze(0).to(self.model.device),
            'attention_mask': build_input_ids['attention_mask'].unsqueeze(0).to(self.model.device),
            'images': [[build_input_ids['images'][0].to(self.model.device).to(torch.bfloat16)]] if self.image is not None else None,
        }
        gen_kwargs = {
            "max_new_tokens": self.max_string_token_length,
            "temperature": self.temperature,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
            "num_return_sequences": 1,
            "stopping_criteria": [
                StringStoppingCriteria(self.tokenizer, len(input_tokens))
            ],
        }
        response = self.model.generate(**inputs, **gen_kwargs)
        torch.cuda.empty_cache()


        # Some models output the prompt as part of the response
        # This removes the prompt from the response if it is present
        if (
            len(response[0]) >= len(input_tokens)
            and (response[0][: len(input_tokens)] == input_tokens).all()
        ):
            response = response[0][len(input_tokens) :]
        if response.shape[0] == 1:
            response = response[0]

        response = self.tokenizer.decode(response, skip_special_tokens=True)

        self.debug("[generate_string]", "|" + response + "|")

        if response.count('"') < 1:
            return response

        return response.split('"')[0].strip()
    
    def generate_boolean(self) -> bool:
        prompt = self.get_prompt() + '"'
        self.debug("[generate_string]", prompt, is_prompt=True)
        build_input_ids = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=prompt,
            images=[self.image] if self.image is not None else None,
            template_version="base",
        )
        inputs = {
            'input_ids': build_input_ids['input_ids'].unsqueeze(0).to(self.model.device),
            'token_type_ids': build_input_ids['token_type_ids'].unsqueeze(0).to(self.model.device),
            'attention_mask': build_input_ids['attention_mask'].unsqueeze(0).to(self.model.device),
            'images': [[build_input_ids['images'][0].to(self.model.device).to(torch.bfloat16)]] if self.image is not None else None,
        }
        output = self.model.forward(**inputs, use_cache=False)
        torch.cuda.empty_cache()
        logits = output.logits[0, -1]

        true_token_id = self.tokenizer.convert_tokens_to_ids("true")
        false_token_id = self.tokenizer.convert_tokens_to_ids("false")

        result = logits[true_token_id] > logits[false_token_id]

        self.debug("[generate_boolean]", result)

        return result.item()
    

    def generate_array(self, item_schema: Dict[str, Any], obj: Dict[str, Any]) -> list:
        for _ in range(self.max_array_length):
            # forces array to have at least one element
            element = self.generate_value(item_schema, obj)
            obj[-1] = element

            obj.append(self.generation_marker)
            input_prompt = self.get_prompt()
            obj.pop()
            build_input_ids = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=input_prompt,
            images=[self.image] if self.image is not None else None,
            template_version="base",
            )
            inputs = {
            'input_ids': build_input_ids['input_ids'].unsqueeze(0).to(self.model.device),
            'token_type_ids': build_input_ids['token_type_ids'].unsqueeze(0).to(self.model.device),
            'attention_mask': build_input_ids['attention_mask'].unsqueeze(0).to(self.model.device),
            'images': [[build_input_ids['images'][0].to(self.model.device).to(torch.bfloat16)]] if self.image is not None else None,
            } # TODO: hardcode of dtype
            output = self.model.forward(**inputs, use_cache=False)
            torch.cuda.empty_cache()
            logits = output.logits[0, -1]


            top_indices = logits.topk(30).indices
            sorted_token_ids = top_indices[logits[top_indices].argsort(descending=True)]

            found_comma = False
            found_close_bracket = False

            for token_id in sorted_token_ids:
                decoded_token = self.tokenizer.decode(token_id)
                if ',' in decoded_token:
                    found_comma = True
                    break
                if ']' in decoded_token:
                    found_close_bracket = True
                    break

            if found_close_bracket or not found_comma:
                break

        return obj

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
        prompts = [self.get_prompt(i) for i in range(len(self.prompts))]
        padded_inputs = self.tokenizer(prompts, return_tensors="pt", padding="longest", truncation=True).to(self.model.device)
        input_ids = padded_inputs["input_ids"]
        attention_mask = padded_inputs["attention_mask"]
        self.debug("[generate_number]", input_ids, is_prompt=True)
        
        # Batch processing
        response = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_number_tokens,
            num_return_sequences=1,
            logits_processor=[self.number_logit_processor],
            stopping_criteria=[
                NumberStoppingCriteria(self.tokenizer, len(input_ids[0]))
            ],
            temperature=temperature or self.temperature,
            do_sample=True,
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
                iterations += 1
                results.append(self.generate_number(temperature=self.temperature * 1.3, iterations=iterations))
        
        return results

    def generate_boolean(self) -> List[bool]:
        prompts = [self.get_prompt(i) for i in range(len(self.prompts))]
        padded_inputs = self.tokenizer(prompts, return_tensors="pt", padding="longest", truncation=True).to(self.model.device)

        input_ids = padded_inputs["input_ids"]
        attention_mask = padded_inputs["attention_mask"]
        self.debug("[generate_boolean]", input_ids, is_prompt=True)

        # Batch processing
        output = self.model(input_ids, attention_mask=attention_mask)
        logits = output.logits[:, -1, :]

        true_token_id = self.tokenizer.convert_tokens_to_ids("true")
        false_token_id = self.tokenizer.convert_tokens_to_ids("false")

        results = (logits[:, true_token_id] > logits[:, false_token_id]).tolist()
        self.debug("[generate_boolean]", results)

        return results

    def generate_string(self) -> List[str]:
        prompts = [self.get_prompt(i) for i in range(len(self.prompts))]
        padded_inputs = self.tokenizer(prompts, return_tensors="pt", padding="longest", truncation=True).to(self.model.device)

        input_ids = padded_inputs["input_ids"]
        attention_mask = padded_inputs["attention_mask"]
        self.debug("[generate_string]", input_ids, is_prompt=True)

        # Batch processing
        response = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_string_token_length,
            num_return_sequences=1,
            temperature=self.temperature,
            do_sample=True,
            stopping_criteria=[
                StringStoppingCriteria(self.tokenizer, len(input_ids[0]))
            ],
            pad_token_id=self.tokenizer.eos_token_id,
        )

        responses = []
        for i, resp in enumerate(response):
            if (
                len(resp) >= len(input_ids[i])
                and (resp[: len(input_ids[i])] == input_ids[i]).all()
            ):
                resp = resp[len(input_ids[i]):]
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

            prompts = [self.get_prompt(i) for i in range(len(self.prompts))]
            padded_inputs = self.tokenizer(prompts, return_tensors="pt", padding="longest", truncation=True).to(self.model.device)

            input_ids = padded_inputs["input_ids"]
            attention_mask = padded_inputs["attention_mask"]

            output = self.model(input_ids, attention_mask=attention_mask)
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
            values = self.generate_value(schema, obj_list, key)
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
                for i in range(len(obj_list)):
                    obj_list[i][key] = self.generation_marker
            else:
                for i in range(len(obj_list)):
                    obj_list[i].append(self.generation_marker)
            return self.generate_number()
        elif schema_type == "boolean":
            if key:
                for i in range(len(obj_list)):
                    obj_list[i][key] = self.generation_marker
            else:
                for i in range(len(obj_list)):
                    obj_list[i].append(self.generation_marker)
            return self.generate_boolean()
        elif schema_type == "string":
            if key:
                for i in range(len(obj_list)):
                    obj_list[i][key] = self.generation_marker
            else:
                for i in range(len(obj_list)):
                    obj_list[i].append(self.generation_marker)
            return self.generate_string()
        elif schema_type == "array":
            new_arrays = [[]*len(obj_list)]
            for i in range(len(obj_list)):
                obj_list[i][key] = new_arrays[i]
            return self.generate_array(schema["items"], new_arrays)
        elif schema_type == "object":
            new_objs = [{} for _ in range(len(obj_list))]
            if key:
                for i in range(len(obj_list)):
                    obj_list[i][key] = new_objs[i]
            else:
                for i in range(len(obj_list)):
                    obj_list[i].append(new_objs[i])
            return self.generate_object(schema["properties"], new_objs)
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    def __call__(self) -> List[Dict[str, Any]]:
        self.value = [{} for _ in range(len(self.prompts))]
        generated_data = self.generate_object(self.json_schema["properties"], self.value)
        return generated_data
               