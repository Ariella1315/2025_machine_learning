import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression

# ---------------------------
# Step 1: Load and preprocess data
# ---------------------------
tree = ET.parse('data.xml')
root = tree.getroot()
ns = {'cwa': 'urn:cwa:gov:tw:cwacommon:0.1'}

# Extract geographic info
geo = root.find('.//cwa:GeoInfo', ns)
lon0 = float(geo.find('cwa:BottomLeftLongitude', ns).text)
lat0 = float(geo.find('cwa:BottomLeftLatitude', ns).text)
lon1 = float(geo.find('cwa:TopRightLongitude', ns).text)
lat1 = float(geo.find('cwa:TopRightLatitude', ns).text)

# Grid resolution
nx, ny = 67, 120
dlon = (lon1 - lon0) / (nx - 1)
dlat = (lat1 - lat0) / (ny - 1)

# Extract temperature values
content = root.find('.//cwa:Content', ns).text.strip()
values = np.array([float(v.replace('E', 'e')) for v in content.replace('\n', ',').split(',')])
grid = values.reshape(ny, nx)



# Build coordinates and mask invalid data
lons = np.linspace(lon0, lon1, nx)
lats = np.linspace(lat0, lat1, ny)
lon_grid, lat_grid = np.meshgrid(lons, lats)
mask = grid != -999.0
X = np.column_stack((lon_grid[mask], lat_grid[mask]))
y = grid[mask]

# ---------------------------
# Step 2: Gaussian Discriminant Analysis (GDA)
# ---------------------------
class GDA:
    def fit(self, X, y):
        self.y0 = y == 0
        self.y1 = y == 1
        self.mu0 = X[self.y0].mean(axis=0)
        self.mu1 = X[self.y1].mean(axis=0)
        self.sigma = ((X[self.y0] - self.mu0).T @ (X[self.y0] - self.mu0) +
                      (X[self.y1] - self.mu1).T @ (X[self.y1] - self.mu1)) / len(X)
        self.sigma_inv = np.linalg.inv(self.sigma)
        self.phi = np.mean(y)

    def predict_prob(self, X):
        def g(mu):
            return np.exp(-0.5 * np.sum((X - mu) @ self.sigma_inv * (X - mu), axis=1))
        g0 = g(self.mu0)
        g1 = g(self.mu1)
        return self.phi * g1 / (self.phi * g1 + (1 - self.phi) * g0)

    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)

# ---------------------------
# Step 3: Classification and k-fold cross-validation
# ---------------------------
#threshold=np.percentile(y, 75)
y_class = (y > 23.5).astype(int)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
acc_scores = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_class[train_idx], y_class[test_idx]

    model = GDA()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = np.mean(y_pred == y_test)
    acc_scores.append(acc)

print(f"Cross-Validation Accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
print("Max temp:", y.max(), "Min temp:", y.min(), "Mean:", y.mean())

# ---------------------------
# Step 4: Plot decision boundary (using last fold)
# ---------------------------
plt.figure(figsize=(6, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', s=10)
plt.title('GDA Classification Decision Boundary (Last Fold)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# ---------------------------
# Step 5: Regression model for warm region
# ---------------------------
R = LinearRegression()
mask_warm = y_class == 1
R.fit(X[mask_warm], y[mask_warm])

# Combined piecewise function
def h(x):
    c_pred = model.predict(x)
    r_pred = R.predict(x)
    return np.where(c_pred == 1, r_pred, -999)

# ---------------------------
# Step 6: Visualization of combined model
# ---------------------------
Z_pred = h(X)
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=Z_pred, cmap='plasma', s=10)
plt.title('Combined Piecewise Model h(x) (GDA + Regression)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

print("Model completed: GDA with 5-fold CV + Regression piecewise function constructed.")
