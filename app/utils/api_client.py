import requests
import streamlit as st
from typing import Dict, List, Any, Optional
import pandas as pd


class NexoCommerceAPIClient:
    """Client for interacting with NexoCommerce API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ML model information"""
        try:
            response = self.session.get(f"{self.base_url}/model/info", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_single(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict single product lifecycle action"""
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=product_data,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict_batch(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict batch of products"""
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json={"products": products},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def multi_agent_analysis(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run multi-agent analysis"""
        try:
            response = self.session.post(
                f"{self.base_url}/analyze",
                json={"products": products},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        try:
            response = self.session.get(f"{self.base_url}/agents/status", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}


@st.cache_resource
def get_api_client(base_url: str = "http://localhost:8000") -> NexoCommerceAPIClient:
    """Get cached API client instance"""
    return NexoCommerceAPIClient(base_url)