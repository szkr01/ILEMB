# ILEMB
## ベクトル表現を利用した直感的な画像検索機
大量の画像をベクトルとして収集し、クエリベクトルと類似するベクトルを列挙する事によって、画像検索を行う
同梱データベースは170万枚の画像情報を記録

## 開発環境
- Python 3.10.11
- Vue3+Typescript

## Installation
### データベース
https://huggingface.co/datasets/szkr/ILEMB/tree/main

### Colabの場合
Google Colab環境で動作するノートブックを作成しました。
Colab_ILEMB.ipynbを読み込み、ノートブック内の説明に従ってください。

170万データセットは無料Colab環境では動きません。


### ローカルの場合
ライブラリインストール
```txt
pip install -r requirments.txt
```
データフレームダウンロード(私のGPUの涙の結晶)
https://huggingface.co/datasets/szkr/ILEMB/resolve/main/databese_1700k.parquet
data/databese_1700k.parquetに設置

CUDAが使用できる場合torchのインストールを工夫してください
CUDAがないと特徴量抽出に非常に時間がかかり、まともに使用できません。(私の環境でGPU使用時で1秒未満、CPU使用時で3分程度)
### 例
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```

## Usage
- app.pyを起動後、以下のURLにアクセス
http://localhost:8002/app

## 検索チュートリアル
タグ、画像を併用して検索が可能です
タグ検索に関して、以下の例を試してみてください
### 良い例
- ```red_hair drill_hair```
- ```red_hair drill_hair (realistic:1.5)```
- ```red_hair drill_hair (realistic:-2)```
### 悪い例
- ```red_hair,drill_hair #不正なスペース```
- ```red_hair,drill_hair (realistic :1.5) #不正なスペース```
- ```akaikami doriru_hea- #存在しないタグ```

### 説明
- タグは空白区切り
- 任意のタグに重みづけをして検索する事が可能
```(tagname:weight)```

- スペースの使い方に気を付ける
- 存在するタグの組み合わせでしか検索できない(補完候補のタグを使用してください)

完全な自然言語から検索するやつは開発中

## Config
環境変数によるコンフィグ設定が可能
App.py Configクラス直書きでもよい

### APP_DATAFRAME_PATH
- 使用するデータフレームのパス

### APP_PORT
- デフォルト 8002
- ポート

### APP_HOST
- デフォルト 0.0.0.0
- ホスト名

### APP_RATING_THRESHOLD
- デフォルト 0
- 0:general, 1:questionable, 2:sensitive, 3:explicit
- nsfw コンテンツを除外するための閾値
- 0 推奨