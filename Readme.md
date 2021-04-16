# 葛山製作所の異常検知
## 1. 仕様
商品の状態を内視鏡で撮影した画像を入力に、以下4種類に分類したい。
- OK  
目立った傷・切削痕も無く、問題なく出荷可能。
- Level 1  
細かな汚れや傷があるが、出荷自体は可能。
- Level 3  
少々目立つ傷や切削痕があり、差し戻し案件。
- Level 5  
出荷不可。最低でもこの状態は判定したい。

## 2. ファイル構成
データセットは `dataset/train/` / `dataset/val` にレベル別で設置してください。  
学習済みモデルはこちらから（[Google drive]()）  
詳細は、金澤（[w68i248a@icloud.com]()）まで。  

## 3. 動作確認環境
- python == 3.7.9
- Pillow == 8.0.0
- tensorboard == 2.2.0
- torch	== 1.6.0
- torchvision == 0.7.0

## 4. 実行
### 画像処理のみでの対応
これから。グレースケールや2値化してからエッジの面積とるとか？

### VGG16を用いた画像分類
※データの前処理はILSVRC2012準拠でやった。
- demo.py : 推論の実行。`$ python demo.py` で動くはず。
- train.py : 訓練の実行。教師データのパスを確認して、`$ python demo.py` で動くはず。

> 結果があまり芳しくない。おそらく前処理の問題が大きいが、転移学習のためにそうせざるを得ない。  
> 1から学習しようにもデータ数的に足りるとは考えにくい。

### 自作モデル組んで画像分類
自作モデル組んでみた。  
条件は **「OKとLevel5を分類できるかどうか」** とした。  
画像に以下の前処理をかけたもので学習・テストし、結果を以下に示す。

> 画像の前処理
>- マスクをかける
>- 810*610で中心をトリミング
>- グレースケール化
>- 大津式二値化をかける

#### 結果
学習データ
- ok : 9472
- ng level 5 : 142

テストデータ
- ok : 2368
- ng level 5 : 36

テスト結果
```
image11832_bin.jpg => predicted: NG-level-5 (true label: ok)
image11834_bin.jpg => predicted: NG-level-5 (true label: ok)
image11835_bin.jpg => predicted: NG-level-5 (true label: ok)
image11836_bin.jpg => predicted: NG-level-5 (true label: ok)
image11837_bin.jpg => predicted: NG-level-5 (true label: ok)
image11838_bin.jpg => predicted: NG-level-5 (true label: ok)
image11839_bin.jpg => predicted: NG-level-5 (true label: ok)
================================
Precision: 0.8372, Recall: 1.0000
OK: 2368 samples, NG-LV5: 36 samples => OK: 2361, NG-LV5: 43 classified
```
