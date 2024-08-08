# JPX-Tokyo-Stock-Exchange-Prediction

https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/code

# 目標

對給定日期的每隻活躍股票進行排名，並根據每日點差回報的夏普比率進行評估

# Kaggle team

![image](https://user-images.githubusercontent.com/48245648/172189051-da1caa8e-d62a-412b-b12f-232c8a936da8.png)

# 最後上傳的Ranking

![image](https://user-images.githubusercontent.com/48245648/172189839-c3c32802-e3b2-4a3b-82ce-2e7cc0382d42.png)

## Feature Engineering
透過對原始資料的分析與處理，使得AI Model可以訓練為更有意義的資料，並且提升Ｍodel的準確性，以下針對我們有使用的Feature介紹
* Upper & Lower shadow:藉由K線的型態、排列方式，了解目前市場情緒，並可作為預測未來是多頭還是空頭的一種指標。K線是由開盤價、最高價、最低價、收盤價所構成，也是記錄買方和賣方實戰的過程；如果收盤價高於開盤價就以「實體紅線」表示，收盤價低於開盤價則以「實體黑線」表示，最高價及最低價則以「影線」表示，高價拉回則留上影線，低價回升則留下影線。
![image](https://user-images.githubusercontent.com/102530486/172061802-61da38a6-c1f1-4b04-8d28-cc67d6947641.png)
* Return:計算與過去股價差距之百分比，如果為正，則為上漲趨勢，反之為負，則下跌趨勢。
* MovingAvg:MA線是一條平滑的曲線，所以可以利用斜率來判斷目前股價的發展趨勢。
* Volatility:波動率高的特點是價格變化節奏極快，交易量較大，市場出現意外重大價格變動。另一方面，波動率較低往往趨於穩定，並且價格波動較小。

```python=
feat_importance['avg'] = feat_importance.mean(axis=1)
feat_importance = feat_importance.sort_values(by='avg',ascending=True)
pal=sns.color_palette("plasma_r", 29).as_hex()[2:]

fig=go.Figure()
for i in range(len(feat_importance.index)):
    fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=feat_importance['avg'][i], 
                       line_color=pal[::-1][i],opacity=0.7,line_width=4))
fig.add_trace(go.Scatter(x=feat_importance['avg'], y=feat_importance.index, mode='markers', 
                         marker_color=pal[::-1], marker_size=8,
                         hovertemplate='%{y} Importance = %{x:.0f}<extra></extra>'))
fig.update_layout(template=temp,title='Overall Feature Importance', 
                  xaxis=dict(title='Average Importance',zeroline=False),
                  yaxis_showgrid=False, margin=dict(l=120,t=80),
                  height=700, width=800)
fig.show()
```
<img width="968" alt="image" src="https://user-images.githubusercontent.com/48245648/172362140-daf901de-9ed6-4728-a075-2321fdc265dd.png">
透過LightGBM所提供的特徵重要性排名，去篩選適合的特徵去訓練模型

## 超參數設定
在實驗過程中，除了Feature的資料可以設計，還有Model的參數可以調整，Optuna 是一個專為機器學習設計的自動超參數優化的框架，透過Optuna調整的超參數來提高模型預測能力。

```python=
def objectives(trial):
    params = {
            'num_leaves': trial.suggest_int('num_leaves', 300, 4000),
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            'max_bin': trial.suggest_int('max_bin', 2, 100),
            'learning_rate': trial.suggest_uniform('learning_rate',0, 1),
    }
    model = LGBMRegressor(**params, device_type = 'gpu')
    model.fit(X_optuna, y)
    score = model.score(X_optuna, y)
    return score
```

## Cross-Validation
為了避免Model過於overfitting，我們設計10-fold Cross-Validation來分割訓練集，每份資料介於2017年至2021年間的任兩個時段。
> 以下為Fold 10的分割 與 驗證完的平均Sharpe ratio
![image](https://user-images.githubusercontent.com/48245648/172364938-a0436e91-d7ef-4bfa-a674-9b4e2b19f42a.png)
