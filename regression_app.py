# 必要なライブラリのインポート
import streamlit as st
import pandas as pd
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
    
    # 予測
    predictions = model.predict(X_test)
    test_data['prediction'] = predictions
    
    df = pd.concat([test_data,train_data], axis=0)

    return df, predictions

# Streamlitアプリケーションのメイン関数
def main():
    st.title("回帰モデルアプリ")

    # ファイルアップロード
    st.write("データのアップロード")
    train_file = st.file_uploader("訓練データをアップロードしてください", type=["csv"])
    test_file = st.file_uploader("テストデータをアップロードしてください", type=["csv"])
    
    # Demoボタンの追加
    st.write("demo: サンプルデータを読み込んで実行")
    demo_button = st.button("Demoを実行")

    # Demoボタンが押されたか、両方のファイルがアップロードされたかをチェック
    if demo_button:
        st.write("サンプルデータを使用してモデルを訓練・実行します...")
        train_file = './train.csv'
        test_file = './test.csv'
    
    if train_file and test_file:
        if not demo_button:  # ボタンが押されていない場合のメッセージ
            st.write("モデルを訓練・実行します...")
        
        df, predictions = perform_regression(train_file, test_file)


        # 予測結果の表示
        st.write("訓練データの実際のターゲット値と予測値:")
        plot_len = len(predictions)+10
        st.line_chart(df[['target','prediction']].iloc[:plot_len])

        # 予測結果をDataFrameに変換
        df_predictions = pd.DataFrame(predictions, columns=["predictions"])

        # 予測結果の表示
        st.write("予測結果:")
        st.write(df_predictions)

        # CSVダウンロードボタンの表示
        csv = df.to_csv(index=False)
        st.download_button("予測結果をCSVとしてダウンロード", data=csv, file_name="predictions.csv", mime="text/csv")

# main関数の実行
if __name__ == "__main__":
    main()
