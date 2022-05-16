import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("star_with_gravity.csv")
df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.columns

radius = df["star_radius"]
mass = df["star_mass"]
gravity = df["Gravity"]

X = df.iloc[:,[3,4]].values

wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(X)
  wcss.append((kmeans.inertia_))


plt.figure(figsize=(10,5))
sns.lineplot(range(1,11),wcss,marker="o",color="red")
plt.title("elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("wcss")
plt.show()


fig = px.scatter(x=radius,y=mass,color=gravity)
fig.show()