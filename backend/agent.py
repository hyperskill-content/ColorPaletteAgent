import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import os
import json
import base64
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BaseColorPaletteAgent(ABC):
    """Abstract base class for color palette extraction agents."""
    
    def __init__(self, n_colors=5):
        self.n_colors = n_colors
    
    def _resize_image(self, img):
        """Resize image if the larger side is > 1024px."""
        width, height = img.size
        max_dim = max(width, height)
        
        if max_dim > 1024:
            scale = 1024 / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return img
    
    @abstractmethod
    def extract_palette(self, image_path: str) -> Dict:
        """
        Extract color palette from image.
        
        Returns:
            Dict with keys: 'image', 'colors', 'metadata'
        """
        pass
    
    @abstractmethod
    def get_agent_name(self) -> str:
        """Return the name of this agent for folder/file naming."""
        pass


class ClassicMLColorPaletteAgent(BaseColorPaletteAgent):
    """Color palette extraction using K-Means clustering."""
    
    def get_agent_name(self) -> str:
        return "classic-ml"
    
    def extract_palette(self, image_path: str) -> Dict:
        """Extract dominant colors using K-Means clustering."""
        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        img = self._resize_image(img)
        img_array = np.array(img)
        
        # Prepare pixels for KMeans
        pixels = img_array.reshape(-1, 3)
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=self.n_colors, n_init='auto', random_state=42)
        kmeans.fit(pixels)
        
        # Get colors (cluster centers) and sort by frequency
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        indices = np.argsort(-counts)
        
        colors = colors[indices]
        percentages = (counts[indices] / len(labels)) * 100
        
        # For K-Means, we don't have specific coordinates, but we can find 
        # representative pixels for each cluster center
        representative_coords = []
        for cluster_idx in indices:
            # Find pixels belonging to this cluster
            cluster_pixels = np.where(labels == cluster_idx)[0]
            if len(cluster_pixels) > 0:
                # Pick a representative pixel (first one)
                pixel_idx = cluster_pixels[0]
                # Convert flat index to 2D coordinates
                y = pixel_idx // img_array.shape[1]
                x = pixel_idx % img_array.shape[1]
                representative_coords.append([x, y])
            else:
                representative_coords.append([0, 0])
        
        # Build result
        result = {
            'image': img,
            'colors': [
                {
                    'rgb': tuple(color),
                    'hex': '#{:02x}{:02x}{:02x}'.format(*color),
                    'percentage': float(pct),
                    'coordinates': coords,
                    'description': f"Color {i+1} ({pct:.1f}%)"
                }
                for i, (color, pct, coords) in enumerate(zip(colors, percentages, representative_coords))
            ],
            'metadata': {
                'agent': self.get_agent_name(),
                'n_colors': self.n_colors,
                'image_size': img.size,
                'method': 'K-Means Clustering'
            }
        }
        
        return result


class SimpleColorPaletteAgent(BaseColorPaletteAgent):
    """Color palette extraction using GPT-4o vision without pixel sampling."""
    
    def __init__(self, n_colors=5):
        super().__init__(n_colors)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment. Please set it in .env file")
        self.client = OpenAI(api_key=api_key)
    
    def get_agent_name(self) -> str:
        return "simple-ai"
    
    def _encode_image(self, img):
        """Encode PIL Image to base64 string."""
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def extract_palette(self, image_path: str) -> Dict:
        """Extract colors using GPT-4o vision."""
        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        img = self._resize_image(img)
        
        # Encode image for API
        base64_image = self._encode_image(img)
        
        # Create prompt
        prompt = f"""Analyze this image and identify the {self.n_colors} most representative colors.
For each color, provide:
1. A hex code (e.g., #FF5733)
2. A brief description of what it represents in the image (e.g., "sky blue", "car red", "grass green")
3. An approximate percentage of how much this color appears in the image

Return ONLY a JSON array in this exact format, with no additional text:
[
  {{"hex": "#RRGGBB", "description": "description here", "percentage": 25.5}},
  ...
]

Focus on the most visually important and representative colors."""

        try:
            # Call GPT-4o
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response (might have markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            colors_data = json.loads(content)
            
            # Convert to our format
            colors = []
            for i, color_info in enumerate(colors_data[:self.n_colors]):
                hex_code = color_info['hex']
                # Convert hex to RGB
                hex_code = hex_code.lstrip('#')
                rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
                
                # For AI-generated colors, we don't have real coordinates
                # Use center of image as placeholder
                coords = [img.size[0] // 2, img.size[1] // 2]
                
                colors.append({
                    'rgb': rgb,
                    'hex': color_info['hex'],
                    'percentage': color_info.get('percentage', 100.0 / len(colors_data)),
                    'coordinates': coords,
                    'description': color_info['description']
                })
            
            result = {
                'image': img,
                'colors': colors,
                'metadata': {
                    'agent': self.get_agent_name(),
                    'n_colors': self.n_colors,
                    'image_size': img.size,
                    'method': 'GPT-4o Vision Analysis'
                }
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract colors using GPT-4o: {str(e)}")


class AdvancedColorPaletteAgent(BaseColorPaletteAgent):
    """
    V3 Agent: GPT-4o generates descriptions -> Qwen-VL localizes pixels -> Actual pixel sampling.
    Follows "No Hex Hallucinations" rule.
    """
    
    def __init__(self, n_colors=5):
        super().__init__(n_colors)
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # OpenRouter client for Qwen-VL
        self.or_api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.or_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment.")
            
        self.or_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.or_api_key,
        )
    
    def get_agent_name(self) -> str:
        return "advanced-v3"
    
    def _encode_image(self, img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def extract_palette(self, image_path: str) -> Dict:
        """Step-by-step extraction: GPT-4o (desc) -> Qwen-VL (coords) -> Sampling."""
        # 1. Load and resize to 1024px standard
        img = Image.open(image_path).convert('RGB')
        img = self._resize_image(img)
        img_array = np.array(img)
        width, height = img.size
        
        # Encode image for APIs
        base64_image = self._encode_image(img)
        
        # 2. GPT-4o: Generate 5 vivid descriptions
        print(f"  > GPT-4o: Generating {self.n_colors} vivid descriptions...")
        desc_prompt = (
            f"Analyze this image and provide {self.n_colors} very specific, vivid descriptions "
            "of DIFFERENT colored objects or areas to sample (e.g., 'the bright red hood of the car', "
            "'the dark moss on the tree trunk'). Return ONLY a JSON list of strings."
        )
        
        desc_response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": desc_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }]
        )
        
        content = desc_response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.split("```")[0].strip()
            
        descriptions = json.loads(content)
        if isinstance(descriptions, dict):
            for val in descriptions.values():
                if isinstance(val, list):
                    descriptions = val
                    break
        descriptions = descriptions[:self.n_colors]
        
        # 3. Qwen-VL: Localize all descriptions using actual pixels in JSON format
        print(f"  > Qwen-VL: Localizing sampling points (Actual Pixels)...")
        
        qwen_prompt = (
            f"The image resolution is {width}x{height}. Identify the exact pixel coordinates [x, y] "
            f"for each of these {len(descriptions)} descriptions: {', '.join(descriptions)}. "
            "Use top-left as [0, 0]. Respond ONLY with a JSON array in this exact schema:\n"
            "[\n"
            "  {\"point_2d\": [x, y], \"label\": \"description\"}\n"
            "]\n"
            "CRITICAL: Use ACTUAL PIXELS based on the resolution provided, NOT normalized 0-1000 values. "
            "Coordinates must be in [x, y] order."
        )
        
        qwen_response = self.or_client.chat.completions.create(
            model="qwen/qwen2.5-vl-32b-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": qwen_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }]
        )
        
        qwen_content = qwen_response.choices[0].message.content.strip()
        print(f"    [DEBUG] Qwen raw output: {qwen_content}")
        
        if qwen_content.startswith("```"):
            qwen_content = qwen_content.split("```")[1]
            if qwen_content.startswith("json"):
                qwen_content = qwen_content[4:]
            qwen_content = qwen_content.split("```")[0].strip()
            
        try:
            points_data = json.loads(qwen_content)
        except json.JSONDecodeError:
            print("    ! Failed to parse Qwen JSON, using fallback parsing.")
            import re
            points_data = []
            # Fallback: find all [x, y] or (x, y) patterns
            matches = re.findall(r'[\[\(](\d+),\s*(\d+)[\]\)]', qwen_content)
            for i, m in enumerate(matches[:len(descriptions)]):
                points_data.append({"point_2d": [int(m[0]), int(m[1])], "label": descriptions[i]})
        
        # 4. Sampling: Get actual RGB from the pixels
        colors = []
        for i, point_info in enumerate(points_data):
            if i >= self.n_colors: break
            
            # Extract point_2d and handle case where it might be y,x from model habits
            coords = point_info.get("point_2d", [0, 0])
            x, y = int(coords[0]), int(coords[1])
            label = point_info.get("label", descriptions[i] if i < len(descriptions) else "color")
            
            # Ensure coordinates are within bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            
            rgb = img_array[y, x]
            hex_code = '#{:02x}{:02x}{:02x}'.format(*rgb)
            
            colors.append({
                'rgb': tuple(rgb.tolist()),
                'hex': hex_code,
                'percentage': 100.0 / self.n_colors,
                'coordinates': [x, y],
                'description': label
            })
            print(f"    [OK] {i+1}. Sampled {label} at [{x}, {y}] -> {hex_code}")
            
        # Ensure we have enough colors
        while len(colors) < self.n_colors:
            colors.append({
                'rgb': (0,0,0), 'hex': '#000000', 'percentage': 0, 
                'coordinates': [0,0], 'description': "Point not localized"
            })
            
        result = {
            'image': img,
            'colors': colors,
            'metadata': {
                'agent': self.get_agent_name(),
                'n_colors': self.n_colors,
                'image_size': img.size,
                'method': 'GPT-4o + Qwen-VL (Pixel-First Spatial Sampling)'
            }
        }
        
        return result


class ExploratoryColorPaletteAgent(BaseColorPaletteAgent):
    """
    V4 Agent: GPT-4o desc -> Qwen-VL rough point -> 60x60 KMeans crop -> GPT-4o vibe pick.
    Extremely robust spatial and semantic reasoning.
    """
    
    def __init__(self, n_colors=5):
        super().__init__(n_colors)
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.or_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY'),
        )
    
    def get_agent_name(self) -> str:
        return "exploratory-v4"
    
    def _encode_image(self, img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def extract_palette(self, image_path: str) -> Dict:
        # 1. Load and resize
        img = Image.open(image_path).convert('RGB')
        img = self._resize_image(img)
        img_array = np.array(img)
        width, height = img.size
        base64_image = self._encode_image(img)
        
        # Step 1: GPT-4o generates vivid targets
        print(f"  > GPT-4o: Generating {self.n_colors} vivid descriptions...")
        desc_prompt = (
            f"Analyze this image and provide {self.n_colors} very specific, vivid descriptions "
            "of DIFFERENT colored objects or areas to sample. Return ONLY a JSON list of strings."
        )
        
        desc_response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": desc_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }]
        )
        
        descriptions = json.loads(desc_response.choices[0].message.content.strip().strip("```json").strip("```"))
        descriptions = descriptions[:self.n_colors]
        
        # Step 2: Qwen-VL gives a rough starting point
        print(f"  > Qwen-VL: Localizing rough points...")
        qwen_prompt = (
            f"The image resolution is {width}x{height}. Identify the exact pixel coordinates [x, y] "
            f"for each of these: {', '.join(descriptions)}. "
            "Respond ONLY with a JSON array: [{\"point_2d\": [x, y], \"label\": \"description\"}]"
        )
        
        qwen_response = self.or_client.chat.completions.create(
            model="qwen/qwen2.5-vl-32b-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": qwen_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }]
        )
        
        qwen_content = qwen_response.choices[0].message.content.strip()
        print(f"    [DEBUG] Qwen raw output: {qwen_content}")
        
        # Robust JSON extraction
        json_content = qwen_content
        if "```" in json_content:
            json_content = json_content.split("```")[-2] # Take content in last code block
            if json_content.startswith("json"):
                json_content = json_content[4:]
            json_content = json_content.strip()
        
        try:
            points_data = json.loads(json_content)
        except json.JSONDecodeError:
            print("    ! Failed to parse Qwen JSON, using fallback parsing.")
            import re
            points_data = []
            matches = re.findall(r'[\[\(](\d+),\s*(\d+)[\]\)]', qwen_content)
            for i, m in enumerate(matches[:len(descriptions)]):
                points_data.append({"point_2d": [int(m[0]), int(m[1])], "label": descriptions[i]})
        
        colors = []
        for i, point_info in enumerate(points_data):
            if i >= self.n_colors: break
            
            x_rough, y_rough = point_info["point_2d"]
            desc = point_info["label"]
            
            # Step 3: Python crops 60x60 and extracts 5 distinct colors via KMeans
            x1 = max(0, x_rough - 30)
            y1 = max(0, y_rough - 30)
            x2 = min(width, x_rough + 30)
            y2 = min(height, y_rough + 30)
            
            crop = img_array[y1:y2, x1:x2]
            pixels = crop.reshape(-1, 3)
            
            kmeans = KMeans(n_clusters=5, n_init='auto', random_state=42)
            kmeans.fit(pixels)
            cluster_centers = kmeans.cluster_centers_.astype(int)
            hex_candidates = ['#%02x%02x%02x' % tuple(c) for c in cluster_centers]
            
            # Step 4: GPT-4o picks the one that matches the vibe
            print(f"  > GPT-4o: Picking best match for '{desc}'...")
            vibe_prompt = (
                f"For the description '{desc}', I extracted these 5 real colors from that area: "
                f"{', '.join(hex_candidates)}. Which ONE of these hex codes matches the 'vibe' best? "
                "Return ONLY the hex code, e.g., #FF0000"
            )
            
            vibe_response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": vibe_prompt}]
            )
            
            chosen_hex = vibe_response.choices[0].message.content.strip().upper()
            if chosen_hex not in [h.upper() for h in hex_candidates]:
                # Fallback to the first one if GPT hallucinates a new hex
                chosen_hex = hex_candidates[0]
            
            # Convert back to RGB
            hex_val = chosen_hex.lstrip('#')
            chosen_rgb = tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))
            
            colors.append({
                'rgb': chosen_rgb,
                'hex': chosen_hex,
                'percentage': 100.0 / self.n_colors,
                'coordinates': [x_rough, y_rough],
                'description': desc
            })
            print(f"    [OK] {i+1}. Vibe match: {chosen_hex} for '{desc}'")
            
        return {
            'image': img,
            'colors': colors,
            'metadata': {
                'agent': self.get_agent_name(),
                'n_colors': self.n_colors,
                'image_size': img.size,
                'method': 'V4: GPT-4o + Qwen-VL + 60x60 KMeans + Vibe Pick'
            }
        }


def get_agent(version: str = "classic-ml", n_colors: int = 5) -> BaseColorPaletteAgent:
    """
    Factory function to get the appropriate agent.
    
    Args:
        version: Agent version ("classic-ml", "simple-ai", etc.)
        n_colors: Number of colors to extract
    
    Returns:
        Instance of BaseColorPaletteAgent subclass
    """
    agents = {
        "classic-ml": ClassicMLColorPaletteAgent,
        "simple-ai": SimpleColorPaletteAgent,
        "advanced-v3": AdvancedColorPaletteAgent,
        "exploratory-v4": ExploratoryColorPaletteAgent,
    }
    
    if version not in agents:
        raise ValueError(f"Unknown agent version: {version}. Available: {list(agents.keys())}")
    
    return agents[version](n_colors=n_colors)

