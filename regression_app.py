# 必要なライブラリのインポート
import streamlit as st
import pandas as pd
import time
from sklearn.linear_model import LinearRegression

# 回帰モデルの関数
def perform_regression(train_file, test_file):
    # データの読み込み
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # 説明変数と目的変数を分離
    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]
    X_test = test_data

    # 線形回帰モデルを訓練
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 訓練データに対する予測値の取得
    train_data['prediction'] = model.predict(X_train)
    train_data['is_test'] = 0
    
    # 予測
    test_data['prediction'] = model.predict(X_test)
    test_data['is_test'] = 1
    
    df = pd.concat([train_data,test_data], axis=0)

    return df

# Streamlitアプリケーションのメイン関数
def main():
    st.title("回帰モデルアプリ")

    # ファイルアップロード
    st.markdown('''
        ## モデル
        - scikit-learnの線形回帰です
        ## データ
        - データに時間軸がある場合、上から「古い→新しい」順にしてください
        - trainにだけ予測する値を列名「target」でいれてください
        - testには「target」列はいれないでください
        - trainとtestのそのほかの列は同じ順番に並べてください
        - trainの「target」以外の列すべてで学習を行います
        ''')
    
    train_file = st.file_uploader("訓練データをアップロードしてください", type=["csv"])
    test_file = st.file_uploader("テストデータをアップロードしてください", type=["csv"])

    st.write("demo: サンプルデータを読込")
    demo_button = st.button("demo")
    if demo_button:
        train_file = './train.csv'
        test_file = './test.csv'
        st.success('data is ready!')

    if not demo_button:
        st.write("データを読み込んで実行ボタンを押してください")
    
    start_button = st.button("学習実行")
    if start_button:
        with st.spinner(text='In progress'):
            df = perform_regression(train_file, test_file)
            time.sleep(1)
        
        st.success('Done!')

        # 予測結果の表示
        st.write("訓練データの実際のターゲット値と予測値:")
        
        st.line_chart(df[['target','prediction']],color=['#6495ED','#e95295'])

        st.write("test 予測結果:")
        st.dataframe(df)

        # CSVダウンロードボタンの表示
        csv = df.to_csv(index=False)
        st.download_button("予測結果をCSVとしてダウンロード", data=csv, file_name="predictions.csv", mime="text/csv")

# main関数の実行
if __name__ == "__main__":
    main()
