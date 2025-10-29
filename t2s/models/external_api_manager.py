"""External API manager for cloud-based AI models."""

import os
import asyncio
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from rich.console import Console


class APIClient(ABC):
    """Base class for external API clients."""

    def __init__(self, api_key: str):
        """Initialize API client."""
        self.api_key = api_key
        self.console = Console()
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> str:
        """Generate completion from the API."""
        pass

    @abstractmethod
    def validate_api_key(self) -> bool:
        """Validate the API key."""
        pass


class AnthropicAPIClient(APIClient):
    """Anthropic Claude API client."""

    def __init__(self, api_key: str):
        """Initialize Anthropic client."""
        super().__init__(api_key)
        self.model_id = "claude-haiku-4-5"
        self.base_url = "https://api.anthropic.com/v1/messages"
        self._client = None

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install it with: pip install anthropic"
                )
        return self._client

    async def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> str:
        """Generate completion from Claude API."""
        try:
            client = self._get_client()

            # Create message with system prompt
            message = await asyncio.to_thread(
                client.messages.create,
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            # Extract text from response
            return message.content[0].text

        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise RuntimeError(f"Claude API error: {str(e)}")

    def validate_api_key(self) -> bool:
        """Validate Anthropic API key."""
        try:
            client = self._get_client()
            # Test with a minimal request
            client.messages.create(
                model=self.model_id,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception as e:
            self.logger.error(f"Anthropic API key validation failed: {e}")
            return False


class XAIAPIClient(APIClient):
    """XAI Grok API client."""

    def __init__(self, api_key: str):
        """Initialize XAI client."""
        super().__init__(api_key)
        self.model_id = "grok-code-fast-1"
        self.base_url = "https://api.x.ai/v1"
        self._client = None

    def _get_client(self):
        """Get or create OpenAI-compatible client for XAI."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install it with: pip install openai"
                )
        return self._client

    async def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> str:
        """Generate completion from Grok API."""
        try:
            client = self._get_client()

            # Use OpenAI-compatible chat completion
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"XAI Grok API error: {e}")
            raise RuntimeError(f"Grok API error: {str(e)}")

    def validate_api_key(self) -> bool:
        """Validate XAI API key."""
        try:
            client = self._get_client()
            # Test with a minimal request
            client.chat.completions.create(
                model=self.model_id,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception as e:
            self.logger.error(f"XAI API key validation failed: {e}")
            return False


class GeminiAPIClient(APIClient):
    """Google Gemini API client."""

    def __init__(self, api_key: str):
        """Initialize Gemini client."""
        super().__init__(api_key)
        self.model_id = "gemini-2.5-flash"
        self._client = None

    def _get_client(self):
        """Get or create Gemini client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. "
                    "Install it with: pip install google-genai"
                )
        return self._client

    async def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> str:
        """Generate completion from Gemini API."""
        try:
            client = self._get_client()

            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # Generate content
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model_id,
                contents=full_prompt,
                config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature
                }
            )

            return response.text

        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            raise RuntimeError(f"Gemini API error: {str(e)}")

    def validate_api_key(self) -> bool:
        """Validate Gemini API key."""
        try:
            client = self._get_client()
            # Test with a minimal request
            client.models.generate_content(
                model=self.model_id,
                contents="Hi",
                config={"max_output_tokens": 10}
            )
            return True
        except Exception as e:
            self.logger.error(f"Gemini API key validation failed: {e}")
            return False


class OpenAIAPIClient(APIClient):
    """OpenAI API client."""

    def __init__(self, api_key: str, model_id: str = "gpt-4o-mini"):
        """Initialize OpenAI client."""
        super().__init__(api_key)
        self.model_id = model_id
        self._client = None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install it with: pip install openai"
                )
        return self._client

    async def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> str:
        """Generate completion from OpenAI API."""
        try:
            client = self._get_client()

            # Use chat completion
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    def validate_api_key(self) -> bool:
        """Validate OpenAI API key."""
        try:
            client = self._get_client()
            # Test with a minimal request
            client.chat.completions.create(
                model=self.model_id,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception as e:
            self.logger.error(f"OpenAI API key validation failed: {e}")
            return False


class ExternalAPIManager:
    """Manager for external API-based models."""

    def __init__(self, config):
        """Initialize external API manager."""
        self.config = config
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        self._clients: Dict[str, APIClient] = {}

    def _get_api_client(self, model_id: str) -> Optional[APIClient]:
        """Get API client for a specific model."""
        # Check if model is an external API model
        if model_id not in self.config.EXTERNAL_API_MODELS:
            return None

        model_config = self.config.EXTERNAL_API_MODELS[model_id]
        provider = model_config.get("provider")
        model_name = model_config.get("name", model_id)

        # Get API key from config
        api_keys = self.config.config.api_keys
        api_key = api_keys.get(provider)

        if not api_key:
            provider_names = {
                "anthropic": "Anthropic",
                "xai": "XAI",
                "google": "Google",
                "openai": "OpenAI"
            }
            provider_display = provider_names.get(provider, provider)
            raise ValueError(
                f"API key not configured for {model_name}.\n"
                f"Please configure your {provider_display} API key:\n"
                f"  1. Run 't2s config'\n"
                f"  2. Select 'External API Keys'\n"
                f"  3. Set {provider_display} API Key"
            )

        # Create or reuse client
        cache_key = f"{provider}:{model_id}"
        if cache_key not in self._clients:
            if provider == "anthropic":
                self._clients[cache_key] = AnthropicAPIClient(api_key)
            elif provider == "xai":
                self._clients[cache_key] = XAIAPIClient(api_key)
            elif provider == "google":
                self._clients[cache_key] = GeminiAPIClient(api_key)
            elif provider == "openai":
                # Extract specific model ID
                openai_model_id = model_config.get("api_model_id", "gpt-4o-mini")
                self._clients[cache_key] = OpenAIAPIClient(api_key, openai_model_id)
            else:
                raise ValueError(f"Unknown provider: {provider}")

        return self._clients[cache_key]

    async def generate_sql(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """Generate SQL using external API."""
        try:
            client = self._get_api_client(model_id)
            if not client:
                raise ValueError(f"Model {model_id} is not an external API model")

            # Generate completion
            response = await client.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1024,
                temperature=0.1
            )

            return response

        except Exception as e:
            self.logger.error(f"Error generating SQL with {model_id}: {e}")
            raise

    def validate_api_key(self, provider: str, api_key: str) -> bool:
        """Validate API key for a provider."""
        try:
            if provider == "anthropic":
                client = AnthropicAPIClient(api_key)
            elif provider == "xai":
                client = XAIAPIClient(api_key)
            elif provider == "google":
                client = GeminiAPIClient(api_key)
            elif provider == "openai":
                client = OpenAIAPIClient(api_key)
            else:
                return False

            return client.validate_api_key()

        except Exception as e:
            self.logger.error(f"API key validation error for {provider}: {e}")
            return False

    def is_api_model(self, model_id: str) -> bool:
        """Check if model is an external API model."""
        return model_id in self.config.EXTERNAL_API_MODELS
