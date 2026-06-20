# 指南：如何生成TUM标志的绘制轨迹

本文档旨在指导用户如何使用 `Tumdrawplanning.py` 脚本和相关的路径点文件，来生成ABB GoFa机器人绘制"TUM"标志的完整运动轨迹。

## 核心概念

绘制过程被分解为三个独立的阶段，每个阶段由一个特定的CSV文件定义：

1.  **第一阶段：接近 (Approach)**
    *   **文件**: `src/planning_input/ABB_GoFa/waypoints2_r_TCP.csv`
    *   **动作**: 机器人工具从安全高度垂直下降到“绘图平面”。

2.  **第二阶段：绘制 (Draw)**
    *   **文件**: `src/planning_input/ABB_GoFa/waypoints3_r_TCP.csv`
    *   **动作**: 机器人在恒定的高度上，沿着文件中定义的23个关键点移动，绘制出TUM标志的精确形状。

3.  **第三阶段：抬起 (Retract)**
    *   **文件**: `src/planning_input/ABB_GoFa/waypoints4_r_TCP.csv`
    *   **动作**: 绘制完成后，机器人工具垂直抬起，返回到安全高度。

## 操作步骤

要生成完整的轨迹，你需要手动修改并**分三次运行** `Tumdrawplanning.py` 脚本。

---

### **第1步: 生成“接近”轨迹**

1.  **修改脚本**:
    *   打开 `src/Tumdrawplanning.py` 文件。
    *   找到以下两行代码：
      ```python
      waypoints_r_TCP__0 = np.genfromtxt(os.path.join('planning_input','ABB_GoFa','waypoints_r_TCP.csv'), delimiter=',')
      waypoints_A_N0= np.genfromtxt(os.path.join('planning_input','ABB_GoFa','waypoints_A_N0.csv'), delimiter=',')
      ```
    *   将它们修改为指向**阶段一**的文件：
      ```python
      waypoints_r_TCP__0 = np.genfromtxt(os.path.join('planning_input','ABB_GoFa','waypoints2_r_TCP.csv'), delimiter=',')
      waypoints_A_N0= np.genfromtxt(os.path.join('planning_input','ABB_GoFa','waypoints2_A_N0.csv'), delimiter=',')
      ```

2.  **运行脚本**:
    *   在命令行中执行: `python src/Tumdrawplanning.py`

3.  **保存结果 (建议)**:
    *   脚本会生成 `q.csv` 和 `trajectory.csv`。建议将它们重命名，以免被后续步骤覆盖，例如：
        *   `q_approach.csv`
        *   `trajectory_approach.csv`

---

### **第2步: 生成“绘制”轨迹**

1.  **修改脚本**:
    *   再次打开 `src/Tumdrawplanning.py`。
    *   将文件名修改为指向**阶段二**的文件：
      ```python
      waypoints_r_TCP__0 = np.genfromtxt(os.path.join('planning_input','ABB_GoFa','waypoints3_r_TCP.csv'), delimiter=',')
      waypoints_A_N0= np.genfromtxt(os.path.join('planning_input','ABB_GoFa','waypoints3_A_N0.csv'), delimiter=',')
      ```

2.  **运行脚本**:
    *   再次执行: `python src/Tumdrawplanning.py`

3.  **保存结果 (建议)**:
    *   将新生成的 `q.csv` 和 `trajectory.csv` 重命名为：
        *   `q_draw.csv`
        *   `trajectory_draw.csv`

---

### **第3步: 生成“抬起”轨迹**

1.  **修改脚本**:
    *   最后一次打开 `src/Tumdrawplanning.py`。
    *   将文件名修改为指向**阶段三**的文件：
      ```python
      waypoints_r_TCP__0 = np.genfromtxt(os.path.join('planning_input','ABB_GoFa','waypoints4_r_TCP.csv'), delimiter=',')
      waypoints_A_N0= np.genfromtxt(os.path.join('planning_input','ABB_GoFa','waypoints4_A_N0.csv'), delimiter=',')
      ```

2.  **运行脚本**:
    *   执行: `python src/Tumdrawplanning.py`

3.  **保存结果 (建议)**:
    *   将最终生成的 `q.csv` 和 `trajectory.csv` 重命名为：
        *   `q_retract.csv`
        *   `trajectory_retract.csv`

---

## 最终成果

完成以上三个步骤后，你就拥有了 `_approach`, `_draw`, `_retract` 三套轨迹文件。将它们按顺序组合，即可得到机器人执行“下降-绘制-抬起”这一完整任务的全部运动数据。
