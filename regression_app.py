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

    # 予測
    predictions = model.predict(X_test)

    return predictions

# Streamlitアプリケーションのメイン関数
def main():
    st.title("回帰モデルアプリ")

    # ファイルアップロード機能を追加
    train_file = st.file_uploader("訓練データをアップロードしてください", type=["csv"])
    test_file = st.file_uploader("テストデータをアップロードしてください", type=["csv"])

    # 両方のファイルがアップロードされた場合、回帰モデルを実行
    if train_file and test_file:
        st.write("モデルを訓練・実行します...")
        predictions = perform_regression(train_file, test_file)
        
        # 予測結果をDataFrameに変換
        df_predictions = pd.DataFrame(predictions, columns=["predictions"])

        # 予測結果の表示
        st.write("予測結果:")
        st.write(df_predictions)

        # CSVダウンロードボタンの表示
        csv = df_predictions.to_csv(index=False)
        st.download_button("予測結果をCSVとしてダウンロード", data=csv, file_name="predictions.csv", mime="text/csv")

# main関数の実行
if __name__ == "__main__":
    main()
