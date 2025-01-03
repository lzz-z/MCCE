import os
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential, get_bearer_token_provider
class LLM:
    def __init__(self,model='chatgpt'):
        self.model_choice = model
        self.model = self._init_model(model)
        self.chat = self._init_chat(model)

    def _init_chat(self,model):
        if model == 'chatgpt':
            return self.gpt_chat

    def _init_model(self,model):
        if model == 'chatgpt':
            return self._init_chatgpt()

    def _init_chatgpt(self):
        # Set the necessary variables
        resource_name = "sfm-openai-sweden-central"#"gcrgpt4aoai2c" sfm-openai-sweden-central  ds-chatgpt4o-ai-swedencentral
        endpoint = f"https://{resource_name}.openai.azure.com/"
        api_version = "2024-02-15-preview"  # Replace with the appropriate API version

        azure_credential = ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential(
                exclude_cli_credential=True,
                # Exclude other credentials we are not interested in.
                exclude_environment_credential=True,
                exclude_shared_token_cache_credential=True,
                exclude_developer_cli_credential=True,
                exclude_powershell_credential=True,
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credentials=True,
                managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
            )
        )

        token_provider = get_bearer_token_provider(azure_credential,
            "https://cognitiveservices.azure.com/.default")
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider
        )
        
        
        return client
    
    def gpt_chat(self, content):
        completion = self.model.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who can propose novel and powerful molecules based on your domain knowledge."},
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )
        res = completion.choices[0].message.content
        return res