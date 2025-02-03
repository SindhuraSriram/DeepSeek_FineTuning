# **Fine-Tuning DeepSeek with Unsloth and Hugging Face**  

## **Overview**  
This Jupyter Notebook provides a comprehensive guide to **fine-tuning the DeepSeek-R1 model** using **Unsloth, Hugging Face, and Weights & Biases (W&B)**. The workflow includes **data preprocessing, model loading, LoRA fine-tuning, evaluation, and tracking** to optimize DeepSeek for custom NLP tasks.  

## **Requirements**  
Ensure you have the following dependencies installed before running the notebook:  

### **Python Packages**
```bash
pip install unsloth datasets transformers peft accelerate bitsandbytes torch trl wandb
```

### **Additional Dependencies**  
- **Hugging Face Transformers**: For model and tokenizer handling.
- **Unsloth**: Efficient LoRA-based fine-tuning framework.
- **Bitsandbytes (bnb)**: 4-bit quantization for memory-efficient training.
- **Weights & Biases (W&B)**: Experiment tracking and logging.

---

## **Workflow Overview**  

### **1️⃣ Setup & Environment Configuration**  
- Install **Unsloth** and upgrade to the latest version.
- Import required libraries (**FastLanguageModel**, **Transformers**, **PEFT**, **TRL**, etc.).
- Set up **GPU acceleration** and **LoRA configurations** for memory-efficient fine-tuning.

### **2️⃣ Loading the Pretrained DeepSeek Model**  
- The **DeepSeek-R1-Distill-Llama-8B** model is loaded using **Unsloth**.
- The tokenizer is initialized, and **LoRA adapters** are applied for fine-tuning.
- The model operates in **4-bit mode (bnb-4bit)** for efficient training.

### **3️⃣ Data Preprocessing & Formatting**  
- **Dataset:** Uses **Alpaca-GPT4** dataset via Hugging Face Datasets library.
- **Standardization:** Converts dataset into **ShareGPT format**.
- **Custom Chat Templates:** Defines instruction-response format for better fine-tuning.

### **4️⃣ Model Fine-Tuning**  
- **Trainer:** Uses **`SFTTrainer` from Hugging Face TRL**.
- **Gradient Checkpointing:** Enabled for memory efficiency.
- **Training Configuration:**
  - **Batch Size:** Adjustable for memory constraints.
  - **LoRA Parameters:** `r=4`, `lora_alpha=16`, `dropout=0`
  - **Random Seed:** Set for reproducibility.

### **5️⃣ Evaluation & Logging with Weights & Biases (W&B)**  
- Monitors training metrics such as **loss, accuracy, and performance trends**.
- Uses **W&B tracking** to visualize learning curves.

### **6️⃣ Saving & Deploying the Fine-Tuned Model**  
- Saves **LoRA-adapted weights** separately.
- Converts back to **full weights for inference**.
- The model can be deployed using **Hugging Face Inference API or Ollama**.

---

## **Usage Instructions**  

### **Running the Notebook**
1. **Clone the repository and navigate to the directory**  
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   jupyter notebook Fine_tune_DeepSeek.ipynb
   ```
2. **Run the cells in sequence**  
   - Modify training hyperparameters as needed.
   - Track metrics using **Weights & Biases**.

3. **Save & Export Fine-Tuned Weights**
   - The final model can be saved and deployed via **Hugging Face or Ollama**.

---

## **Customization**  
- Modify **LoRA settings** for different task-specific fine-tuning.
- Adjust **dataset formats and preprocessing steps**.
- Tune **hyperparameters** for better model convergence.

## **Troubleshooting & Notes**  
- **Ensure GPU acceleration** (`torch.cuda.is_available()`) is enabled for optimal performance.
- **Adjust LoRA configurations** to prevent out-of-memory errors.
- **Use W&B for experiment tracking** and hyperparameter optimization.

---

## **Author**  
📧 **Sindhura Sriram**  
🔗 **[LinkedIn](https://www.linkedin.com/in/sindhura-sriram/)**  
📂 **[GitHub](https://github.com/SindhuraSriram)**  
🌐 **[Portfolio](https://sindhura-sriram.com/)**  
