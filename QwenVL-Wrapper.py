import base64
import io
import os
import tempfile
from PIL import Image
from langchain_core.language_models.llms import LLM
import gc
from typing import Any, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

class QwenLLM(LLM):
    """
      Downloading and Installing Qwen VL - Vision Model:
    """
    model_name: str = "Qwen/Qwen-VL-Chat-Int4"
    model_type: str = "vision"

    inference_type: str = None
    model: Any = None
    tokenizer: Any = None

    def __init__(self, infer_type="chat", **kwargs):
        super(QwenLLM, self).__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.inference_type = infer_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="cuda", trust_remote_code=True).eval()

    @property
    def _llm_type(self) -> str:
        return "Vision Model"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n":"hello"}
    
    def destruct_model_Inferencing(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    def _call(
        self,
        prompt: str,
        image: Optional[str] = None,
        history: Optional[List[str]] = None,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.inference_type == "vision":
          import imghdr
          if image is None:
              raise ValueError("image kwargs is required")
          image_data = base64.b64decode(image)

          image_format = imghdr.what(None, h=image_data)
          if not image_format:
              raise ValueError("Invalid image format")

          with tempfile.NamedTemporaryFile(suffix=f".{image_format}", delete=False) as temp_file:
              temp_file.write(image_data)
              temp_file_path = temp_file.name

          query = self.tokenizer.from_list_format([
              {'image': temp_file_path},
              {'text': prompt.strip()},
          ])

          response = self.model.chat(self.tokenizer, query=query, history=history)
          print(response)
          if stop is not None:
              raise ValueError("stop kwargs are not permitted.")
          return str(response)

        if self.inference_type == "chat":
          query = self.tokenizer.from_list_format([
              {'text': prompt.strip()},
          ])
          response = self.model.chat(self.tokenizer, query=query, history=history)
          print(response)
          return str(response)
