# Análise e Classificação do Dataset Seeds

Este projeto automatiza a classificação de grãos de trigo em três variedades usando aprendizado de máquina. O projeto é dividido em vários notebooks para organização e compreensão mais claras:

## Estrutura dos Notebooks

1. **01-Exploracao-Dados.ipynb**
   - Importação e exploração inicial do dataset.
   - Visualização e estatísticas descritivas.

2. **02-Preprocessamento.ipynb**
   - Limpeza e normalização dos dados.
   - Divisão em treino e teste para os modelos.

3. **03-Treinamento-Modelos.ipynb**
   - Implementação de três algoritmos de aprendizado de máquina:
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
     - Random Forest
   - Avaliação inicial dos modelos.

4. **04-Otimizacao-Hiperparametros.ipynb**
   - Otimização de hiperparâmetros usando GridSearchCV.
   - Avaliação dos modelos ajustados.

5. **05-Insights-e-Conclusoes.ipynb**
   - Análise dos resultados e comparação dos modelos.
   - Insights principais e recomendações.

---

## Detalhes de Cada Notebook

### 01-Exploracao-Dados.ipynb

#### Objetivo:
Compreender a estrutura e propriedades do conjunto de dados.

#### Etapas Principais:
1. Carregar o dataset usando Pandas.
2. Exibir as primeiras linhas para visualizar a estrutura e os tipos de dados.
3. Calcular estatísticas descritivas (média, mediana e desvio padrão).
4. Visualizar os dados usando:
   - Histogramas para análise de distribuição.
   - Boxplots para identificar outliers.
   - Matriz de correlação para observar relações entre as variáveis.

---

### 02-Preprocessamento.ipynb

#### Objetivo:
Preparar o conjunto de dados para o treinamento dos modelos.

#### Etapas Principais:
1. Verificar e corrigir valores ausentes (nenhum encontrado).
2. Normalizar as variáveis usando StandardScaler.
3. Dividir o conjunto de dados em **70% para treinamento** e **30% para teste**.

---

### 03-Treinamento-Modelos.ipynb

#### Objetivo:
Treinar e avaliar os modelos de aprendizado de máquina.

#### Modelos Implementados:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Random Forest**

#### Etapas Principais:
1. Treinar cada modelo usando os dados de treino.
2. Avaliar o desempenho com base nos dados de teste.
3. Métricas utilizadas:
   - Acurácia
   - Precision, Recall e F1-Score
   - Matriz de confusão

#### Resultados Iniciais:
| Modelo          | Acurácia |
|-----------------|----------|
| KNN             | 90.48%   |
| SVM             | 92.06%   |
| Random Forest   | 88.89%   |

---

### 04-Otimizacao-Hiperparametros.ipynb

#### Objetivo:
Aprimorar o desempenho dos modelos através da otimização de hiperparâmetros.

#### Método Utilizado:
- **GridSearchCV**: Busca sistemática pelos melhores parâmetros para cada modelo.

#### Melhores Parâmetros Encontrados:
| Modelo         | Melhores Parâmetros                   |
|----------------|---------------------------------------|
| KNN            | `n_neighbors=5, weights=uniform`     |
| SVM            | `C=1, kernel=linear`                 |
| Random Forest  | `max_depth=None, n_estimators=50`    |

#### Resultados Após Otimização:
| Modelo          | Acurácia |
|-----------------|----------|
| KNN             | 90.48%   |
| SVM             | 90.48%   |
| Random Forest   | 88.89%   |

---

### 05-Insights-e-Conclusoes.ipynb

#### Insights Principais:
1. **Melhor Modelo:**
   - O **SVM** e o **KNN** obtiveram a maior acurácia (90.48%).
2. **Importância das Variáveis:**
   - A variável **Área**, seguida de **Perímetro** e **Comprimento_Nucleo**, mostrou-se fundamental para a classificação.
3. **Random Forest:**
   - Apesar de um desempenho ligeiramente inferior, fornece insights úteis sobre a importância das features.

#### Recomendação:
O modelo **SVM** é recomendado para implementação devido à sua precisão consistente e robustez.

---

## Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/classificacao-seeds.git
   ```
2. Navegue até a pasta do projeto:
   ```bash
   cd classificacao-seeds
   ```
3. Abra os notebooks em um ambiente Jupyter ou Google Colab.

