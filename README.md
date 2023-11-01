# regression_app_demo
streamlitによる回帰アプリのデモです。
## train, test データ形式
- CSV (UTF-8)であること
- 時系列の場合、上から「古い→新しい」であること
- train
    - 予測する変数を「target」という列名にすること
    - 「target」以外の列をすべて特徴量として利用します
- test
    - trainと同じ順番で同じ数の特徴量が入っていること
    - 「target」は存在しないこと
