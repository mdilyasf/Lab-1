# Laboratory Practical: Kalman Filtering for Human Motion Tracking

## Objectives
By the end of this practical session, students should be able to:
1. Understand the fundamental principles of Kalman filtering
2. Implement a Kalman filter for position estimation
3. Analyze the effectiveness of Kalman filtering in noise reduction
4. Apply Kalman filtering to real-world motion tracking scenarios

## Duration
- 3 hours

## Prerequisites
- Basic Python programming
- Understanding of linear algebra and probability
- Familiarity with NumPy and Matplotlib libraries

## Required Software
- Python 3.x
- NumPy
- Matplotlib

---

## 1. Theoretical Background

### 1.1 The Kalman Filter
The Kalman filter is a recursive algorithm that uses a series of measurements observed over time, containing statistical noise and other inaccuracies, to produce estimates of unknown variables that tend to be more accurate than those based on a single measurement alone.

### 1.2 Key Concepts
- **State Vector (x)**: Contains the variables we want to estimate
- **State Transition Matrix (A)**: Describes how the state evolves
- **Measurement Matrix (H)**: Relates the state to measurements
- **Process Noise Covariance (Q)**: Represents uncertainty in the process
- **Measurement Noise Covariance (R)**: Represents uncertainty in measurements
- **Error Covariance Matrix (P)**: Represents uncertainty in state estimates

### 1.3 Kalman Filter Equations
The algorithm consists of two main steps:

1. **Prediction Step**:
   ```
   x̂ₖ₋ = Ax̂ₖ₋₁
   Pₖ₋ = APₖ₋₁Aᵀ + Q
   ```

2. **Update Step**:
   ```
   Kₖ = Pₖ₋Hᵀ(HPₖ₋Hᵀ + R)⁻¹
   x̂ₖ = x̂ₖ₋ + Kₖ(zₖ - Hx̂ₖ₋)
   Pₖ = (I - KₖH)Pₖ₋
   ```

---

## 2. Implementation Walkthrough

### 2.1 Class Definition and Initialization
```python
class KalmanFilter:
    def __init__(self, dt, process_variance, measurement_variance):
```

**Key Parameters:**
- `dt`: Time step between measurements (seconds)
- `process_variance`: Uncertainty in the motion model (Q)
- `measurement_variance`: Uncertainty in sensor measurements (R)

**Matrix Initialization:**
```python
self.A = np.array([[1, dt],
                   [0, 1]])
```
- State transition matrix models constant velocity motion
- First row: position = previous_position + velocity * dt
- Second row: velocity = previous_velocity

```python
self.H = np.array([[1, 0]])
```
- Measurement matrix extracts position from state vector
- We only measure position, not velocity

### 2.2 Process and Measurement Noise
```python
self.Q = np.array([[0.25*dt**4, 0.5*dt**3],
                   [0.5*dt**3, dt**2]]) * process_variance
```
- Q matrix derived from physics of constant acceleration
- Terms account for position and velocity uncertainty

```python
self.R = np.array([[measurement_variance]])
```
- R matrix represents sensor measurement uncertainty
- Single value as we only measure position

### 2.3 Prediction Method
```python
def predict(self):
    self.x = np.dot(self.A, self.x)
    self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
```
- Projects state ahead in time using motion model
- Updates uncertainty considering process noise

### 2.4 Update Method
```python
def update(self, measurement):
    S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
    K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
    y = measurement - np.dot(self.H, self.x)
    self.x = self.x + np.dot(K, y)
    self.P = np.dot((I - np.dot(K, self.H)), self.P)
```
- Computes Kalman gain (K)
- Updates state estimate using measurement
- Updates uncertainty based on measurement reliability

---

## 3. Practical Exercises

### Exercise 1: Basic Implementation
1. Copy and run the provided code
2. Observe the generated plots
3. Calculate and record the improvement in accuracy:
   ```python
   improvement = (measured_error - filtered_error) / measured_error * 100
   print(f"Improvement in accuracy: {improvement:.1f}%")
   ```

### Exercise 2: Parameter Experimentation
Modify the following parameters and observe their effects:

1. Process Variance:
   ```python
   process_variance = [0.01, 0.1, 1.0]  # Try each value
   ```

2. Measurement Variance:
   ```python
   measurement_variance = [0.1, 0.3, 0.9]  # Try each value
   ```

3. Time Step:
   ```python
   dt = [0.05, 0.1, 0.2]  # Try each value
   ```

Record your observations in a table:
| Parameter | Value | Effect on Tracking | Effect on Smoothing |
|-----------|-------|-------------------|---------------------|
|           |       |                   |                     |

### Exercise 3: Analysis Questions
1. How does increasing process variance affect the filter's response to sudden changes?
2. What happens when measurement variance is set very low? Very high?
3. How does the time step affect the filter's performance?
4. What are the trade-offs between responsiveness and smoothing?

---

## 4. Walking Data Generation

### 4.1 Understanding the Synthetic Data
```python
def generate_walking_data(num_steps, dt):
    true_velocity = 1.4  # average walking speed (m/s)
    time = np.arange(num_steps) * dt
    
    # Random accelerations
    accelerations = np.random.normal(0, 0.1, num_steps)
    
    # True positions with acceleration
    true_positions = true_velocity * time + \
                    np.cumsum(0.5 * accelerations * dt**2)
    
    # Add measurement noise
    measured_positions = true_positions + \
                        np.random.normal(0, 0.3, num_steps)
```

**Key Components:**
1. Constant velocity (1.4 m/s) represents average walking speed
2. Random accelerations model natural speed variations
3. Position calculated using physics equations
4. Gaussian noise added to simulate sensor measurements

---

## 5. Extended Exercises

### Exercise 4: Real-Time Simulation
Modify the code to process measurements one at a time:
```python
position = 0
for t in range(num_steps):
    # Generate new measurement
    measurement = measured_positions[t]
    
    # Process with Kalman filter
    kf.predict()
    filtered_pos = kf.update(measurement)
    
    # Real-time plotting (optional)
    plt.clf()
    plt.plot(...)
    plt.pause(0.1)
```

### Exercise 5: Performance Metrics
Implement additional error metrics:
1. Root Mean Square Error (RMSE)
2. Maximum Absolute Error
3. Standard Deviation of Error

---

## 6. Discussion Questions

1. In what scenarios would Kalman filtering be most beneficial?
2. What assumptions does this implementation make about human walking motion?
3. How could this implementation be improved for more realistic scenarios?
4. What are the limitations of using a constant velocity model?

---

## 7. Submission Requirements

1. Complete code implementation with comments
2. Plots showing results from all exercises
3. Filled parameter experimentation table
4. Answers to analysis and discussion questions
5. Brief report (max 2 pages) discussing findings

---

## References

1. Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems
2. Welch, G. & Bishop, G. An Introduction to the Kalman Filter
3. Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. Estimation with Applications to Tracking and Navigation

