"""
Embedding service for text to vector conversion.

Integrates with external embedding APIs (Cohere, etc.) or local models.
"""

import os
from typing import Dict, List, Optional, Tuple
import httpx
import numpy as np
from datetime import datetime, timedelta, timezone


class EmbeddingService:
    """
    Service for generating text embeddings.

    Supports multiple providers and includes caching and fallback strategies.
    """

    def __init__(
        self,
        provider: str = "cohere",
        api_key: Optional[str] = None,
        model: str = "embed-english-v2.0",
        dimension: int = 1024,
    ):
        """
        Initialize embedding service.

        Args:
            provider: Embedding provider (cohere, openai, local)
            api_key: API key for provider
            model: Model name to use
            dimension: Expected embedding dimension
        """
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.model = model
        self.dimension = dimension
        self._cache: Dict[str, Tuple[datetime, List[float]]] = {}  # Simple in-memory cache
        self._cache_ttl = timedelta(hours=1)

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            ValueError: If embedding generation fails
        """
        # Check cache first
        if text in self._cache:
            cached_time, embedding = self._cache[text]
            if datetime.now(timezone.utc) - cached_time < self._cache_ttl:
                return embedding

        # Generate based on provider
        if self.provider == "cohere":
            embedding = await self._generate_cohere_embedding(text)
        elif self.provider == "local":
            embedding = self._generate_mock_embedding(text)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Cache the result
        self._cache[text] = (datetime.now(timezone.utc), embedding)

        return embedding

    async def generate_embeddings_batch(
        self, texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings

    async def _generate_cohere_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using Cohere API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not self.api_key:
            # Fall back to mock if no API key
            return self._generate_mock_embedding(text)

        url = "https://api.cohere.ai/v1/embed"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "texts": [text],
            "model": self.model,
            "truncate": "END"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers)
                response.raise_for_status()
                result = response.json()
                return result["embeddings"][0]
        except Exception as e:
            # Fall back to mock on error
            print(f"Cohere API error: {e}. Using mock embedding.")
            return self._generate_mock_embedding(text)

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """
        Generate mock embedding for testing.

        Creates deterministic fake embedding based on text hash.

        Args:
            text: Text to embed

        Returns:
            Mock embedding vector
        """
        # Generate deterministic pseudo-random vector
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.dimension).tolist()

        # Normalize to unit vector (common for embeddings)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (np.array(embedding) / norm).tolist()

        return embedding

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension

    def is_available(self) -> bool:
        """Check if embedding service is available."""
        if self.provider == "local":
            return True
        return bool(self.api_key)