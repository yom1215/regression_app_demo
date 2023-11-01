# 必要なライブラリのインポート
import streamlit as st
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 

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

    start_button = False

    start_button = st.button("学習・推論実行")
    
    # Demoボタン
    # stateの初期化
    if "demo_clicked" not in st.session_state:
        st.session_state.demo_clicked = False

    st.subheader("demo: サンプルデータを読み込んで実行")
    demo_button = st.button("Demoを実行", type="primary")

    if demo_button:
        st.session_state.demo_clicked = True

    # Demoボタンが押されたかをチェック
    if st.session_state.demo_clicked:
        st.write("サンプルデータを読み込みます...")
        train_file = './train.csv'
        test_file = './test.csv'
        start_button = True
        
    if start_button:
        st.write("モデルを訓練・実行します...")

        with st.spinner(text='In progress'):
            df = perform_regression(train_file, test_file)
            time.sleep(1)
            
        st.success('Done!')

        # 予測結果の表示
        st.subheader("学習結果の表示")
        st.write("訓練データの実際のtargetの値と予測値:")
        st.caption("グラフ上でスクロールすると拡大できます")
        
        st.line_chart(df[['target','prediction']],color=['#6495ED','#e95295'])

        st.scatter_chart(data=df, x='target', y='prediction')

        mse = mean_squared_error(y_true = df.dropna()['target'], y_pred=df.dropna()['prediction'])

        st.write(f"train MSE : {mse}")

        st.write("test 予測結果:")
        st.dataframe(df[df.is_test==1])

        # CSVダウンロードボタンの表示
        csv = df.to_csv(index=False)
        st.download_button("予測結果をCSVとしてダウンロード", data=csv, file_name="predictions.csv", mime="text/csv")

# main関数の実行
if __name__ == "__main__":
    main()
