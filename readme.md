# ILEMB：ベクトル表現を利用した直感的な画像検索エンジン

ILEMBは、大量の画像をベクトルとして格納し、クエリベクトルと類似するベクトルを検索することで、高速な画像検索を実現するシステムです。同梱データベースには170万枚の画像情報が記録されています。

## 特徴

*   **高速な画像検索:** ベクトル間の類似度計算により、高速な検索を実現。
*   **直感的な操作:** タグと画像を組み合わせた検索が可能。
*   **大規模データベース:** 170万枚の画像情報を収録 (ダウンロード提供)。
*   **柔軟な検索:** タグの重み付けによる検索結果の調整。
*   **NSFWフィルタ:** レーティング閾値による不適切なコンテンツの除外。

## 開発環境

*   Python 3.10.11
*   Vue3 + TypeScript

## インストール方法

### 1. データベース

*   以下のURLからデータベースをダウンロードしてください。
    [https://huggingface.co/datasets/szkr/ILEMB/tree/main](https://huggingface.co/datasets/szkr/ILEMB/tree/main)

### 2. 実行環境

#### Google Colab (推奨)

*   `Colab_ILEMB.ipynb` をGoogle Colabで開き、ノートブック内の指示に従って実行してください。
*   **注意:** 170万件のデータセットは、無料のColab環境では動作しない場合があります。

#### ローカル環境

1.  **ライブラリのインストール:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **データフレームのダウンロード:**
    [https://huggingface.co/datasets/szkr/ILEMB/resolve/main/database_1700k.parquet](https://huggingface.co/datasets/szkr/ILEMB/resolve/main/database_1700k.parquet)
    から `database_1700k.parquet` をダウンロードし、`data/` ディレクトリに配置してください。
3.  **CUDA (GPU) の設定 (オプション):**
    *   CUDAが利用可能な場合は、PyTorchのインストールをCUDA対応版に変更してください。これにより、特徴量抽出が大幅に高速化されます。
    *   **例 (CUDA 11.8 の場合):**
        ```bash
        pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
        ```
    *   CUDAを使用しない場合、特徴量抽出に非常に時間がかかります（CPU使用時はGPU使用時に比べ大幅に遅くなります。例：GPUで1秒未満、CPUで3分程度）。

## 使い方

1.  `app.py` を実行します。
2.  ブラウザで `http://localhost:8002/app` にアクセスします。

## 検索チュートリアル

タグと画像を組み合わせて検索できます。タグ検索の際は、以下の点に注意してください。

### 良い例：

*   `red_hair drill_hair` (タグは空白区切り)
*   `red_hair drill_hair (realistic:1.5)` (タグに重み付け)
*   `red_hair drill_hair (realistic:-2)` (タグに負の重み付け)

### 悪い例：

*   `red_hair,drill_hair` (カンマ区切りは不可)
*   `red_hair,drill_hair (realistic :1.5)` (重み付けのコロン前後にスペースは不要)
*   `akaikami doriru_hea-` (存在しないタグ)

### タグ検索のポイント:

*   タグは**スペース**で区切ります。
*   ` (tagname:weight) ` の形式で、特定のタグに重み付けできます。
*   **スペースの使い方**に注意してください。
*   **存在するタグ**の組み合わせでのみ検索できます (補完候補のタグを使用してください)。

**自然言語検索機能は開発中です。**

## 設定 (Config)

環境変数または `app.py` 内の `Config` クラスを直接編集することで設定を変更できます。

*   **`APP_DATAFRAME_PATH`**: 使用するデータフレームのパス。
*   **`APP_PORT`**: ポート番号 (デフォルト: 8002)。
*   **`APP_HOST`**: ホスト名 (デフォルト: 0.0.0.0)。
*   **`APP_RATING_THRESHOLD`**: NSFWコンテンツを除外するための閾値 (デフォルト: 0)。
    *   0: General (全年齢対象)
    *   1: Questionable
    *   2: Sensitive
    *   3: Explicit
    *   **推奨値: 0**