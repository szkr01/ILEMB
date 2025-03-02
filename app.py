from typing import List, Optional, Tuple
import io
import time
import os

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import faiss
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
import timm
from torch import nn
from dataclasses import dataclass

from fastapi.middleware.cors import CORSMiddleware

# ==================== 設定 ====================

class Config:
    """アプリケーションの設定"""
    APP_REPO_NAME = os.environ.get("APP_REPO_NAME", "SmilingWolf/wd-eva02-large-tagger-v3")
    APP_FAISS_INDEX_TYPE = os.environ.get("APP_FAISS_INDEX_TYPE", "IndexFlatIP") 
    
    APP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    APP_DATAFRAME_PATH = os.environ.get("APP_DATAFRAME_PATH", "data/databese_1700k.parquet")
    
    APP_PORT = int(os.environ.get("APP_PORT", 8002))
    APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
    
    APP_GET_IMAGES_LIMIT = int(os.environ.get("APP_GET_IMAGES_LIMIT", 2000))
    APP_RATING_THRESHOLD = int(os.environ.get("APP_RATING_THRESHOLD", 0))
    # nsfw コンテンツを除外するための閾値
    # 0:general, 1:questionable, 2:sensitive, 3:explicit

config = Config()

# ==================== データモデル ====================

@dataclass
class ImageEntry:
    """画像エントリのデータモデル"""
    id: str
    url: str
    media_url: str
    rating: int

# ==================== ユーティリティ ====================

class URLConverter:
    """URLを効率的な表現にするクラス"""
    EXTENSION_MAPPING = {"jpg": 0, "png": 1, "jpeg": 2, "bmp": 3, "webp": 4}
    REVERSE_EXTENSION_MAPPING = {v: k for k, v in EXTENSION_MAPPING.items()}
    CDN_PREFIX = "https://cdn.donmai.us/720x720/"
    FILENAME_LENGTH = 32

    def _validate_url(self, url: str):
        if not isinstance(url, str):
            raise ValueError("URL must be a string.")
        if not url.startswith(self.CDN_PREFIX):
            raise ValueError("Invalid URL format.")

    def _validate_numerical(self, numerical_representation: Tuple[np.uint8, np.uint8, np.uint64, np.uint64, np.uint8]):
        if not isinstance(numerical_representation, tuple) or len(numerical_representation) != 5:
            raise ValueError("Input must be a tuple of length 5.")
        dir1, dir2, filename_part1, filename_part2, extension_code = numerical_representation
        if not isinstance(dir1, np.uint8) or not isinstance(dir2, np.uint8) or \
           not isinstance(filename_part1, np.uint64) or not isinstance(filename_part2, np.uint64) or \
           not isinstance(extension_code, np.uint8):
            raise ValueError("Tuple elements have incorrect types.")

    def encode(self, url: str) -> Tuple[np.uint8, np.uint8, np.uint64, np.uint64, np.uint8]:
        """URLを数値表現にエンコードする"""
        self._validate_url(url)
        parts = url.split("/")
        dir1 = np.uint8(int(parts[4], 16))
        dir2 = np.uint8(int(parts[5], 16))
        filename_with_ext = parts[6]
        filename_str, ext_str = filename_with_ext.split(".")

        if len(filename_str) != self.FILENAME_LENGTH:
            raise ValueError(f"Filename is not {self.FILENAME_LENGTH} characters long.")

        filename_part1 = np.uint64(int(filename_str[:16], 16))
        filename_part2 = np.uint64(int(filename_str[16:], 16))

        if ext_str not in self.EXTENSION_MAPPING:
            raise ValueError(f"Unsupported extension: {ext_str}")
        extension = np.uint8(self.EXTENSION_MAPPING[ext_str])

        return (dir1, dir2, filename_part1, filename_part2, extension)

    def decode(self, numerical_representation: Tuple[np.uint8, np.uint8, np.uint64, np.uint64, np.uint8]) -> str:
        """数値表現をURLにデコードする"""
        self._validate_numerical(numerical_representation)
        dir1, dir2, filename_part1, filename_part2, extension_code = numerical_representation

        filename_part1_hex = format(filename_part1, "016x")
        filename_part2_hex = format(filename_part2, "016x")
        filename_hex = filename_part1_hex + filename_part2_hex
        extension_str = self.REVERSE_EXTENSION_MAPPING[extension_code]

        return f"{self.CDN_PREFIX}{dir1:02x}/{dir2:02x}/{filename_hex}.{extension_str}"

# ==================== 画像処理 ====================

class ImageProcessor:
    """画像を前処理するクラス"""
    TARGET_SIZE = 448
    PADDING_COLOR = 255

    def preprocess_image(self, img: Image.Image) -> torch.Tensor:
        _, _, h, w = img.size()
        aspect_ratio = w / h

        if aspect_ratio > 1:
            new_w = self.TARGET_SIZE
            new_h = int(self.TARGET_SIZE / aspect_ratio)
        else:
            new_h = self.TARGET_SIZE
            new_w = int(self.TARGET_SIZE * aspect_ratio)

        transform = transforms.Compose([transforms.Resize((new_h, new_w))])
        resized_img = transform(img)
        padding_left = (self.TARGET_SIZE - new_w) // 2
        padding_top = (self.TARGET_SIZE - new_h) // 2
        padding_right = self.TARGET_SIZE - new_w - padding_left
        padding_bottom = self.TARGET_SIZE - new_h - padding_top
        padding = (padding_left, padding_top, padding_right, padding_bottom)
        padded_img = pad(resized_img, padding, self.PADDING_COLOR)
        padded_img = padded_img[:, [2, 1, 0]]

        return padded_img.div(255.0)

class FeatureExtractor:
    """画像から特徴量を抽出するクラス"""
    def __init__(self, repo_name: str, device: str):
        print("Initializing feature extractor...")
        self.device = device
        self.model: nn.Module = timm.create_model(f"hf-hub:{repo_name}").eval()
        state_dict = timm.models.load_state_dict_from_hf(f"{repo_name}")
        self.model.load_state_dict(state_dict)

        model_head = self.model.head.state_dict()
        self.tag_feature = model_head["weight"].cpu().numpy().astype(np.float16)
        self.tag_feature_bias = model_head["bias"].cpu().numpy().astype(np.float16)

        self.model.head = nn.Identity()
        self.model = self.model.to(device, dtype=torch.float16)

    def extract_feature(self, image: torch.Tensor) -> np.ndarray:
        """画像リストから特徴量を抽出する"""
        img_batch = image.to(self.device, dtype=torch.float16)

        with torch.no_grad():
            pred = self.model(img_batch)
            return pred.cpu().numpy().astype(np.float16)

    def extract_feature_from_tag_index(self, index: int) -> np.ndarray:
        baias_mult = 1 + (np.arctan(self.tag_feature_bias[index])+np.pi/2)/np.pi
        normalized_feature = self.tag_feature[index] / np.linalg.norm(self.tag_feature[index])
        return normalized_feature * baias_mult

# ==================== Tag処理 ====================

class TagProcessor:
    """タグを処理するクラス"""
    def __init__(self, repo_name: str, tag_file: str):
        print("Initializing tag processor...")
        self.tag_df = pd.read_csv(f"https://huggingface.co/{repo_name}/raw/main/{tag_file}")
        self.tag_df = self.tag_df.sort_values("count", ascending=False)

    def get_tags(self, prefix: str) -> List[Tuple[str, int, int]]:
        if not prefix:
            return []
        tag_entries = [(row["name"], row["category"], row["count"])
                       for index, row in self.tag_df.iterrows() if row["name"].startswith(prefix)]
        return tag_entries

    def tag_to_index(self, tag: str) -> Optional[int]:
        if tag not in self.tag_df["name"].values:
            return None
        return self.tag_df[self.tag_df["name"] == tag].index[0]

    def str_to_tags(self, string: str) -> List[Tuple[int, float]]:
        """空白で区切られた文字列をタグに変換する（重み付き）"""
        tags_with_weights = []

        parts = string.split()
        for part in parts:
            part = part.strip()
            if part.startswith("(") and part.endswith(")") and ":" in part:
                try:
                    tag, weight_str = part[1:-1].split(":", 1)
                    tag = tag.strip()
                    weight = float(weight_str)
                except ValueError:
                    tag = part.strip()
                    weight = 1.0
            else:
                tag = part.strip()
                weight = 1.0

            tag_index = self.tag_to_index(tag)
            if tag_index is not None:
                tags_with_weights.append((tag_index, weight))
        print(tags_with_weights)
        return tags_with_weights

# ==================== Faiss インデックス ====================

class FaissIndex:
    """Faiss インデックスを管理するクラス"""
    def __init__(self, embeddings: np.ndarray, index_type: str):
        print("Initializing Faiss index...")
        self.embeddings = embeddings
        self.dimension = embeddings.shape[1]
        self.index = self._build_index(index_type)
        self.index.add(self.embeddings)
        print(f"Index built with {self.embeddings.shape[0]} entries")

    def _build_index(self, index_type: str):
        print(f"Building Faiss index of type: {index_type}")
        if index_type == "IndexFlatL2":
            return faiss.IndexFlatL2(self.dimension)
        elif index_type == "IndexFlatIP":
            return faiss.IndexFlatIP(self.dimension)
        elif index_type == "IndexHNSWFlat":
            return faiss.IndexHNSWFlat(self.dimension, 32)  # M=32 は一例
        elif index_type == "IndexIVFFlat":
            nlist = 100  # クラスタ数 (データ量に応じて調整)
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            index.train(self.embeddings)  # トレーニングが必要
            return index
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def search(self, feature: np.ndarray, limit: int) -> Tuple[np.ndarray, np.ndarray]:
        """類似画像を検索する"""
        D, I = self.index.search(np.float16(feature), limit)
        return D, I

# ==================== API ====================

class ImageSearchAPI:
    """画像検索APIのメインクラス"""
    def __init__(self, config: Config):
        print("Initializing API...")
        self.config = config
        self.app = FastAPI()
        self.url_converter = URLConverter()
        self.tag_processor = TagProcessor(config.APP_REPO_NAME, "selected_tags.csv")
        self.feature_extractor = FeatureExtractor(config.APP_REPO_NAME, config.APP_DEVICE)
        print("Loading dataframe...")
        self.df = pd.read_parquet(config.APP_DATAFRAME_PATH)
        
		#rating processing
        self.df = self.df[self.df["rating"] <= config.APP_RATING_THRESHOLD].reset_index(drop=True)

        self.faiss_index = FaissIndex(np.array(self.df["emb"].tolist()), config.APP_FAISS_INDEX_TYPE)
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # ガバガバ
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.app.mount("/assets", StaticFiles(directory="web/assets"), name="static")
        self.app.mount("/app", StaticFiles(directory="web", html=True), name="frontend")

        self._setup_routes()

    def _setup_routes(self):
        self.app.add_api_route("/API/search", self.search_images, methods=["POST"], response_model=List[ImageEntry])
        self.app.add_api_route("/API/images", self.get_images, methods=["GET"], response_model=List[ImageEntry])
        self.app.add_api_route("/API/images/search_by_image", self.search_by_image, methods=["POST"], response_model=List[ImageEntry])
        self.app.add_api_route("/API/tags", self.get_tags_endpoint, methods=["GET"], response_model=List[Tuple[str, int, int]])
        self.app.add_api_route("/API/images/{image_id}/similar", self.search_by_id, methods=["GET"], response_model=List[ImageEntry])

    async def search_images(
            self,
            q: Optional[str] = None,
            image: Optional[UploadFile] = File(None),
            limit: Optional[int] = Form(default=config.APP_GET_IMAGES_LIMIT),
        ):
        """
        アップロードされた画像に基づいて類似画像を検索して返す。
        """
        feature = np.zeros(self.faiss_index.dimension, dtype=np.float16).reshape(1, -1)

        if image is not None:
            self._validate_image_type(image.content_type)
            image_data = await image.read()
            img = Image.open(io.BytesIO(image_data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            img_tensor = torch.tensor(np.array(img), dtype=torch.float16).permute(2, 0, 1).unsqueeze(0)
            processor = ImageProcessor()
            processed_image = processor.preprocess_image(img_tensor)
            image_feature = self.feature_extractor.extract_feature(processed_image)
            normalized_image_feature = image_feature / np.linalg.norm(image_feature)
            feature += normalized_image_feature
        
        if q is not None:
            tags = self.tag_processor.str_to_tags(q)
            features = self._extract_features_from_tags(tags)
            print(np.linalg.norm(np.sum(features, axis=0).reshape(1, -1)))
            feature += np.sum(features, axis=0).reshape(1, -1)

        return self.search_by_features(feature, limit)

    async def get_images(self, q: str, limit: int = Form(config.APP_GET_IMAGES_LIMIT)):
        """
        指定されたクエリに合致する画像を検索して返す。
        """
        print(f"Searching images with query: {q} (limit: {limit})")
        tags = self.tag_processor.str_to_tags(q)
        features = self._extract_features_from_tags(tags)

        if not features.size:
            return []

        feature = np.sum(features, axis=0)
        feature = feature.reshape(1, -1)

        return self.search_by_features(feature, limit)

    def _extract_features_from_tags(self, tags_with_weights: List[Tuple[int, float]]) -> np.ndarray:
        features = []
        for tag_index, weight in tags_with_weights:
            feature = self.feature_extractor.extract_feature_from_tag_index(tag_index)
            features.append(feature * weight)
        return np.array(features)

    async def search_by_image(self, image: UploadFile = File(...), limit: int = Form(config.APP_GET_IMAGES_LIMIT)):
        """
        アップロードされた画像に基づいて類似画像を検索して返す。
        """
        self._validate_image_type(image.content_type)

        start_time = time.time()
        try:
            image_data = await image.read()
            img = Image.open(io.BytesIO(image_data))
            img = img.convert("RGB")
            img_tensor = torch.tensor(np.array(img), dtype=torch.float16).permute(2, 0, 1).unsqueeze(0)

            processor = ImageProcessor()
            processed_image = processor.preprocess_image(img_tensor)
            feature = self.feature_extractor.extract_feature(processed_image)

            similar_images = self.search_by_features(feature, limit)

        except Exception as e:
            print(f"Error processing uploaded image: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")
        finally:
            elapsed_time = time.time() - start_time
            print(f"Elapsed time: {elapsed_time:.4f} seconds")

        return similar_images

    def _validate_image_type(self, content_type: str):
        if content_type not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(status_code=400, detail="Invalid image type")

    async def get_tags_endpoint(self, prefix: str):
        """
        指定されたプレフィックスで始まるタグのリストを返す。
        """
        tags = self.tag_processor.get_tags(prefix)
        return tags[:20]

    def get_features_by_id(self, image_id: np.uint64) -> np.ndarray:
        """指定されたIDの画像の特徴量を取得する"""
        try:
            features = self.df.loc[self.df["id"] == image_id, "emb"].iloc[0]
            return np.float16(features).reshape(1, -1)
        except (IndexError, KeyError):
            raise HTTPException(status_code=404, detail=f"Image with id '{image_id}' not found")

    def search_by_features(self, features: np.ndarray, limit: int) -> List[ImageEntry]:
        """指定された特徴量を用いて類似画像を検索する"""

        D, I = self.faiss_index.search(features, limit)
        return self._build_image_entries_from_indices(I[0])

    def _build_image_entries_from_indices(self, indices: np.ndarray) -> List[ImageEntry]:
        similar_images = []
        for i in indices:
            try:
                numerical_representation = tuple(self.df.loc[i, ["url_c1", "url_c2", "url_c3", "url_c4", "url_c5"]])
                decoded_media_url = self.url_converter.decode(numerical_representation)
                entry_id = self.df.loc[i, "id"]
                similar_images.append(ImageEntry(
                    id=str(entry_id),
                    url=f"https://danbooru.donmai.us/posts/{entry_id}",
                    media_url=decoded_media_url,
                    rating=int(self.df.loc[i, "rating"])
                ))
            except (ValueError, KeyError) as e:
                print(f"Error processing image at index {i}: {e}")
        return similar_images

    async def search_by_id(self, image_id: str, limit: int = Form(config.APP_GET_IMAGES_LIMIT)):
        """指定されたIDの画像に類似する画像を検索する"""
        try:
            uint64_image_id = np.uint64(image_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid image ID format")
        features = self.get_features_by_id(uint64_image_id)
        return self.search_by_features(features, limit)

api = ImageSearchAPI(config)

if __name__ == "__main__":
    uvicorn.run(api.app, host=config.APP_HOST, port=config.APP_PORT)