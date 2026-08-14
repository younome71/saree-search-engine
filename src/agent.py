import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from src.search import SareeSearcher


load_dotenv()


MODEL_NAME = "gemini-3.5-flash"


class SareeAgent:

    def __init__(self, image_path=None):

        self.image_path = (
            Path(image_path)
            if image_path
            else None
        )

        self.searcher = SareeSearcher()

        self.client = genai.Client()

        self.tool_declaration = {
            "type": "function",
            "name": "search_similar_sarees",
            "description": (
                "Search the saree catalogue for visually "
                "similar sarees using the currently uploaded "
                "image. Use this when the user asks to find "
                "similar, matching, or alternative sarees "
                "based on the uploaded image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "Number of similar sarees to return. "
                            "Use between 1 and 10."
                        ),
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": []
            }
        }

    # ========================================================
    # TOOL IMPLEMENTATION
    # ========================================================

    def search_similar_sarees(self, top_k=10):

        if self.image_path is None:

            return {
                "success": False,
                "error": (
                    "No image has been uploaded. "
                    "Please upload an image first."
                )
            }

        if not self.image_path.exists():

            return {
                "success": False,
                "error": (
                    "The uploaded image could not be found."
                )
            }

        top_k = max(
            1,
            min(int(top_k), 10)
        )

        results = self.searcher.search(
            str(self.image_path),
            top_k=top_k
        )

        clean_results = []

        for result in results:

            clean_results.append({
                "image_id": int(result["image_id"]),
                "image_filename": str(
                    result["image_filename"]
                ),
                "name": str(result["name"]),
                "sku": str(result["sku"]),
                "score": float(result["score"]),
                "image_url": str(result["image_url"]),
                "website_url": str(
                    result["website_url"]
                ),
                "retail_price": float(
                    result["retail_price"]
                ) if result["retail_price"] is not None else None,
                "discounted_price": float(
                    result["discounted_price"]
                ) if result["discounted_price"] is not None else None
            })

        return {
            "success": True,
            "query_image": self.image_path.name,
            "results": clean_results
        }

    # ========================================================
    # CHAT
    # ========================================================

    def chat(self, user_message):

        system_prompt = """
You are TailorTalk, an AI fashion search assistant.

Your primary purpose is to help users find visually
similar sarees from a fashion catalogue.

You have access to one tool:

search_similar_sarees

Use the tool when the user asks to:
- find similar sarees
- find sarees like the uploaded image
- show visually similar sarees
- find alternatives to the uploaded saree
- search for matching sarees based on the uploaded image

If the user asks for visual similarity and an image is
available, use the search tool.

For general conversation that does not require visual
search, answer normally without using the tool.

When presenting search results:
- Be concise.
- Do not invent product information.
- Use the returned product names.
- Mention that the score represents visual similarity.
- Do not claim that a result is an exact match unless
  the result actually has a score of 1.0.
"""

        prompt = (
            f"{system_prompt}\n\n"
            f"Current uploaded image: "
            f"{self.image_path.name if self.image_path else 'None'}\n\n"
            f"User: {user_message}"
        )

        # ----------------------------------------------------
        # First LLM turn
        # ----------------------------------------------------

        interaction = self.client.interactions.create(
            model=MODEL_NAME,
            input=prompt,
            tools=[
                self.tool_declaration
            ],
            store=True
        )

        # ----------------------------------------------------
        # Find function calls
        # ----------------------------------------------------

        function_calls = [
            step
            for step in interaction.steps
            if step.type == "function_call"
        ]

        # No tool call
        if not function_calls:

            return {
                "message": interaction.output_text,
                "results": []
            }

        # ----------------------------------------------------
        # Execute tool calls
        # ----------------------------------------------------

        all_results = []

        for call in function_calls:

            arguments = call.arguments or {}

            top_k = arguments.get(
                "top_k",
                10
            )

            tool_result = (
                self.search_similar_sarees(
                    top_k=top_k
                )
            )

            if tool_result.get("success"):

                all_results.extend(
                    tool_result.get(
                        "results",
                        []
                    )
                )

            # ------------------------------------------------
            # Send result back using previous interaction ID
            # ------------------------------------------------

            final_interaction = (
                self.client.interactions.create(
                    model=MODEL_NAME,
                    previous_interaction_id=(
                        interaction.id
                    ),
                    input=[
                        {
                            "type": "function_result",
                            "name": call.name,
                            "call_id": call.id,
                            "result": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        tool_result
                                    )
                                }
                            ]
                        }
                    ],
                    tools=[
                        self.tool_declaration
                    ],
                    store=True
                )
            )

        return {
            "message": final_interaction.output_text,
            "results": all_results
        }