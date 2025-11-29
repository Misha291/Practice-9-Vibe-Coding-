"""
Задание 2: Анализ датасета Wine - ШАБЛОН
Цель: Загрузить датасет Wine, провести анализ целевой переменной и признаков
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """Загрузить датасет Wine и конвертировать в DataFrame"""
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    # Заменяем числовые метки на названия классов
    df['target'] = df['target'].map({i: name for i, name in enumerate(data.target_names)})
    return df


def target_analysis(df):
    """Анализ целевой переменной (классы вин)"""
    print("\n=== АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ ===")
    value_counts = df['target'].value_counts()
    print("Количество образцов по классам:")
    print(value_counts)
    print("\nПроцентное соотношение:")
    print((value_counts / len(df) * 100).round(2))


def feature_statistics(df):
    """Вычислить статистику по признакам"""
    print("\n=== СТАТИСТИКА ПО ПРИЗНАКАМ ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('target', errors='ignore')
    stats = pd.DataFrame({
        'mean': df[numeric_cols].mean(),
        'median': df[numeric_cols].median(),
        'std': df[numeric_cols].std(),
        'range': df[numeric_cols].max() - df[numeric_cols].min()
    })
    print(stats.round(3))


def visualize_target(df):
    """Визуализировать распределение целевой переменной"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Столбчатая диаграмма
    df['target'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Распределение классов вин')
    axes[0].set_xlabel('Класс')
    axes[0].set_ylabel('Количество')
    axes[0].tick_params(axis='x', rotation=45)

    # Круговая диаграмма
    df['target'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Процентное распределение классов')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.savefig('02_wine_target_distribution.png', dpi=150)
    plt.close()


def visualize_features(df):
    """Визуализировать распределение признаков"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('target', errors='ignore')
    first_six = numeric_cols[:6]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i, col in enumerate(first_six):
        df[col].hist(bins=20, ax=axes[i], color='lightgreen', edgecolor='black')
        axes[i].set_title(f'Гистограмма: {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Частота')

    plt.tight_layout()
    plt.savefig('02_wine_features_distribution.png', dpi=150)
    plt.close()


def features_by_target(df):
    """Boxplot признаков по классам вин"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('target', errors='ignore')
    first_six = numeric_cols[:6]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes = axes.flatten()

    for i, col in enumerate(first_six):
        sns.boxplot(data=df, x='target', y=col, ax=axes[i])
        axes[i].set_title(f'Boxplot: {col} по классам')
        axes[i].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig('02_wine_features_by_class.png', dpi=150)
    plt.close()


def correlation_analysis(df):
    """Анализ корреляций"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('target', errors='ignore')
    corr_matrix = df[numeric_cols].corr()

    print("\n=== МАТРИЦА КОРРЕЛЯЦИИ (первые 6 признаков) ===")
    print(corr_matrix.iloc[:6, :6].round(2))

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Матрица корреляции признаков')
    plt.tight_layout()
    plt.savefig('02_wine_correlation_matrix.png', dpi=150)
    plt.close()


def main():
    """Главная функция"""
    print("=" * 60)
    print("ЗАДАНИЕ 2: EXPLORATORY DATA ANALYSIS - WINE DATASET")
    print("=" * 60)

    df = load_data()
    print(f"\nДатасет загружен. Размер: {df.shape}")
    print("\nПервые 5 строк:")
    print(df.head())

    target_analysis(df)
    feature_statistics(df)
    visualize_target(df)
    visualize_features(df)
    features_by_target(df)
    correlation_analysis(df)

    print("\n" + "=" * 60)
    print("Анализ завершен!")
    print("=" * 60)


if __name__ == "__main__":
    main()