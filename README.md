# gakunai

## 自身で準部してローカルに準備出ておくもの
- 学内の画像データ
- OSNetの学習済み重み("https://github.com/KaiyangZhou/deep-person-reid"　のModelZooでdownloadする)
### 転移学習なし
sweep_and_test_time.py

### 転移学習あり
test_arc_OT_kNN.py

### 歩行者OD表の分析
analyze_OD_.py

実装環境設定
- pytorchは各自GPUに対応したものを使用
- pyファイルは元データと同じディレクトリに配置
  
