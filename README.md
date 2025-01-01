# PhysioTech: Gesture Control System for Physiotherapy Rehabilitation

## 1. Overview of the Project
PhysioTech is an innovative machine learning project that aims to automate and enhance physiotherapy rehabilitation through gesture recognition. The system uses IoT edge computing and advanced machine learning algorithms to identify and assess the correctness of physiotherapy movements, enabling remote rehabilitation and independent exercise monitoring.

---

## 2. Problem Statement
Traditional physiotherapy requires patients to be physically present at hospitals or clinics, which can be inconvenient and time-consuming. Many patients struggle with mobility issues due to injuries, strokes, or other disabilities, making regular clinic visits challenging. There is a need for an assistive technology that can help patients perform exercises correctly at home while receiving accurate feedback on their movements.

---

## 3. Importance of Addressing this Problem
The significance of this solution is supported by several key statistics and research findings:
- WHO reports 17 million people experience strokes annually, requiring physiotherapy for recovery
- Global aging population expected to increase from 703 million (2019) to 1.5 billion (2050)
- Studies show telerehabilitation services effectively deliver care to patients in remote areas
- Post-operative physiotherapy significantly reduces complications and speeds recovery
- Exercise therapy improves physical functioning and quality of life for arthritis patients

---

## 4. User Group
Primary Users:
- Patients with limited mobility or physical impairments
- Post-surgery recovery patients
- Chronic pain patients (e.g., arthritis)
- Elderly individuals requiring physical therapy
- Athletes recovering from injuries

Secondary Users:
- Physiotherapists monitoring patient progress
- Healthcare providers managing remote care

---

## 5. Dataset and Quality of Raw Data
The project utilizes the IntelliRehabDS dataset, which contains:
- 2,590 entries classified into 9 labels of body gestures
- Data from 29 participants (15 rehabilitation patients, 14 healthy individuals)
- 3:D positions of 25 joints recorded using Microsoft Kinect One sensor
- 30 frames per second capture rate

![Recorded joints visualization from IntelliRehabDS](./doc_image_assests/recorded_joints_in_dataset.png)
*Recorded joints visualization from IntelliRehabDS*

Dataset Quality Characteristics:
- Highly relevant for rehabilitation movement analysis
- Well-chosen features covering key body joint positions
- Reliable labeling with distinct gesture classifications
- Credible source (published on MDPI data)
- Validated by data collection group
- Collected from a rehabilitation centre in Malaysia, representing similar demographic conditions

---

## 6. Model Development and Analysis

### 6.1 Baseline Models

#### 6.1.1 Decision Tree Classifier

**Analysis and Evidence**
- Simple yet effective approach for movement classification
- Capable of handling high-dimensional data
- Interpretable decision-making process

**Data Preprocessing and Feature Engineering**
- One-hot encoding for position features (stand, stand-frame, sit, chair, wheelchair)
![One-Hot Encoding for Class Labels](./doc_image_assests/baseline_one_hot_encode.png)
*One-Hot Encoding for Class Labels*

- Averaging of 10 consecutive instances to reduce dataset size and noise
- Original dataset size reduced from ~135,000 to manageable size

**Hyperparameter Tuning**
- Explored maximum depth range up to 25
- Tested different split criteria (entropy, gini, log_loss)
- Optimal parameters: criterion = log_loss, max_depth = 19

**Evaluation and Performance**
| Metric | Score |
|--------|--------|
| Accuracy | 0.90 |
| Precision | 0.896 |
| Recall | 0.896 |
| F1-score | 0.896 |

![Plot for Accuracy vs Decision Tree Depth; Box and Whisker Plot for Accuracy vs Criterion](./doc_image_assests/baseline_dt_evaluation_graphs.png)
*Plot for Accuracy vs Decision Tree Depth; Box and Whisker Plot for Accuracy vs Criterion*

#### 6.1.2 K-Nearest Neighbors

**Analysis and Evidence**
- Non-parametric approach suitable for movement classification
- Effective for complex decision boundaries
- Distance-based classification appropriate for spatial data

**Data Preprocessing and Feature Engineering**
- Same preprocessing as Decision Tree
- Standardization of numerical features

**Hyperparameter Tuning**
- Tested different numbers of neighbors
- Compared 'uniform' vs 'distance' weights
- Distance-based weights showed superior performance

**Evaluation and Performance**
| Metric | Score |
|--------|--------|
| Accuracy | 0.92 |
| Precision | 0.92 |
| Recall | 0.92 |
| F1-score | 0.92 |

![Plot for Accuracy vs Decision Tree Depth; Box and Whisker Plot for Accuracy vs Criterion](./doc_image_assests/baseline_knn_evaluation_graphs.png)
*Plot for Accuracy vs Number of Neighbours; Box and Whisker Plot for Accuracy vs Weights*

### 6.2 Improved Model - Multi-Layer Perceptron (MLP)

**Analysis and Evidence**
- Deep learning approach capable of learning complex patterns
- Flexible architecture adaptable to varying input sizes
![MLP Model Architecture](./doc_image_assests/improved_model_architecture.png)
*MLP Model Architecture*
- Efficient handling of high-dimensional data

**Data Preprocessing and Feature Engineering**
- Removed position feature for improved robustness
- Implemented sliding window approach (window size: 15, jump factor: 5)
![Demonstration of Sliding Window](./doc_image_assests/demonstration_of_sliding_window.png)
*Demonstration of Sliding Window*

- SMOTE oversampling for class balance
![Graph for each Class and Observed Class imbalance](./doc_image_assests/action_vs_count_graph_class_imbalances.png)
*Graph for each Class and Observed Class imbalance*
- Input dimension: 1125 (25 joints × 3 coordinates × 15 instances)

**Hyperparameter Tuning**
- Tested layer sizes: [512, 640, 768, 896, 1024]
- Learning rates: [0.01, 0.001, 0.0001]
- Optimal architecture: [1024, 896, 896] with learning rate 0.0001

**Evaluation and Performance**
| Metric | Score |
|--------|--------|
| Accuracy | 0.94 |
| Precision | 0.94 |
| Recall | 0.94 |
| F1-score | 0.94 |

![Graph for Model Accuracy; Graph for Model Loss](./doc_image_assests/improved_model_accuracy_and_model_loss_graphs.png)
*Graph for Model Accuracy; Graph for Model Loss*

### 6.3 State-of-the-Art Models

#### 6.3.1 LSTM

**Analysis and Evidence**
- Specialized architecture for sequential data
- Capable of learning long-term dependencies
- Preserves temporal information in movement sequences

**Data Preprocessing and Feature Engineering**
- Reshaped input into 15 timesteps × 75 features
- Maintained temporal relationship between frames
- No flattening required, preserving sequential information

**Hyperparameter Tuning**
- LSTM units: 32-1024 (powers of 2)
- Dense layer units variation
- Dropout rates: 0.2, 0.5
- Optimal configuration results: 
| LSTM Units | Dense Units 1 | Dense Units 2 | Dropout Rate | Loss | Accuracy |
|------------|---------------|---------------|--------------|------|-----------|
| 256 | 128 | 128 | 0.2 | 0.144434 | 94.53% |
| 512 | 256 | 256 | 0.2 | 0.146976 | 94.64% |
| 512 | 512 | 256 | 0.2 | 0.145146 | 94.66% |

Selected configuration: LSTM(512), Dense(512, 256), dropout(0.2)

**Evaluation and Performance**
| Metric | Score |
|--------|--------|
| Training Accuracy | 97.33% |
| Test Accuracy | 98.38% |
| Micro-Average F1 | 0.9838 |
| Macro-Average F1 | 0.9840 |

![Graph for Model Accuracy; Graph for Model Loss](./doc_image_assests/sota_model_lstm_eval_graphs.png)
*Graph for Accuracy of Hyperparameter; Graph for Loss of Hyperparameters*

#### 6.3.2 Transfer Learning (Enhanced LSTM)

**Analysis and Evidence**
- Built upon pre-trained LSTM model
- Added additional dense layer for improved performance
- Focus on reducing training time while maintaining accuracy

**Data Preprocessing and Feature Engineering**
- Utilized same preprocessing as LSTM
- Additional dense layer integration

**Hyperparameter Tuning**
- Dense units: [64, 128]
- Activation functions: [sigmoid, tanh, relu]
- Optimal configuration: dense_units=128, activation='sigmoid'

**Evaluation and Performance**
| Metric | Score |
|--------|--------|
| Training Accuracy | 97.84% |
| Test Accuracy | 98.55% |
| Micro-Average F1 | 0.9855 |
| Macro-Average F1 | 0.9857 |

![Graph for Model Accuracy; Graph for Model Loss](./doc_image_assests/sota_model_transfer_learning_eval_graphs.png)
*Graph for Accuracy of Hyperparameter; Graph for Loss of Hyperparameters*

---

## 7. Model Comparison

Performance Progression:
1. Baseline Models:
   - Decision Tree: 90% accuracy
   - KNN: 92% accuracy
2. Improved Model (MLP): 94% accuracy
3. SOTA Models:
   - LSTM: 98.38% accuracy
   - Transfer Learning: 98.55% accuracy

Key Improvements:
- Significant accuracy increase from baseline to SOTA models
- Better temporal information preservation in LSTM models
- Reduced training time with transfer learning approach
- Improved robustness and generalization with advanced architectures

## 8. Conclusion
PhysioTech demonstrates the successful application of machine learning in healthcare, achieving high accuracy in movement recognition for physiotherapy rehabilitation. The progression from baseline models to SOTA approaches shows significant improvements in performance, with the final transfer learning model achieving 98.55% accuracy. While the technology shows great promise for remote physiotherapy applications, considerations such as internet connectivity and gesture interpretation accuracy need to be addressed for practical implementation.

The project successfully meets its objectives of:
- Accurate movement recognition
- Potential for remote therapy delivery
- Improved accessibility for patients
- Efficient training and deployment capabilities

---
