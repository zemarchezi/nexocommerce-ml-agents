import pandas as pd
import numpy as np
from pathlib import Path
import os

class MarketplaceDataLoader:
    """Carrega e prepara dados de marketplace para análise"""
    
    def __init__(self, data_path='data/raw'):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def download_kaggle_dataset(self, dataset_name):
        """
        Download dataset do Kaggle
        Requer: pip install kaggle
        E configuração de ~/.kaggle/kaggle.json
        """
        try:
            import kaggle
            kaggle.api.dataset_download_files(
                dataset_name,
                path=self.data_path,
                unzip=True
            )
            print(f"✅ Dataset baixado: {dataset_name}")
        except Exception as e:
            print(f"❌ Erro ao baixar dataset: {e}")
            print("💡 Baixe manualmente de Kaggle e coloque em data/raw/")
    
    def load_counterfeit_dataset(self, file_path=None):
        """Carrega dataset de produtos falsificados"""
        if file_path is None:
            file_path = self.data_path / 'counterfeit_products.csv'
        
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Dataset carregado: {len(df)} registros")
            print(f"📊 Colunas: {list(df.columns)}")
            return df
        except FileNotFoundError:
            print(f"❌ Arquivo não encontrado: {file_path}")
            print("💡 Instruções para download:")
            print("1. Acesse: https://www.kaggle.com/datasets/aimlveera/counterfeit-product-detection-dataset")
            print("2. Baixe o arquivo counterfeit_products.csv")
            print(f"3. Coloque em: {self.data_path}")
            return None
    
    def load_alternative_ecommerce_data(self):
        """
        Carrega datasets alternativos de e-commerce
        Opções populares do Kaggle:
        - Brazilian E-Commerce Public Dataset by Olist
        - Amazon Sales Dataset
        - Retail Data Analytics
        """
        datasets = {
            'olist': 'olistbr/brazilian-ecommerce',
            'amazon': 'karkavelrajaj/amazon-sales-dataset',
        }
        
        print("📦 Datasets alternativos disponíveis:")
        for name, path in datasets.items():
            print(f"  - {name}: {path}")
        
        return datasets
    
    def prepare_for_lifecycle_analysis(self, df):
        """
        Transforma qualquer dataset de e-commerce para análise de ciclo de vida
        Cria features necessárias para o modelo
        """
        
        # Identificar colunas disponíveis
        cols = df.columns.str.lower()
        
        # Mapeamento flexível de colunas
        column_mapping = {}
        
        # Identificar ID do produto
        for col in ['product_id', 'productid', 'id', 'sku', 'item_id']:
            if col in cols:
                column_mapping['product_id'] = df.columns[cols.tolist().index(col)]
                break
        
        # Identificar preço
        for col in ['price', 'unit_price', 'product_price', 'amount']:
            if col in cols:
                column_mapping['price'] = df.columns[cols.tolist().index(col)]
                break
        
        # Identificar categoria
        for col in ['category', 'product_category', 'category_name', 'type']:
            if col in cols:
                column_mapping['category'] = df.columns[cols.tolist().index(col)]
                break
        
        print(f"🔍 Mapeamento de colunas identificado: {column_mapping}")
        
        # Criar DataFrame processado
        processed_df = df.copy()
        
        # Renomear colunas identificadas
        if column_mapping:
            processed_df = processed_df.rename(columns={
                v: k for k, v in column_mapping.items()
            })
        
        # Criar features sintéticas se não existirem
        if 'product_id' not in processed_df.columns:
            processed_df['product_id'] = [f'PROD_{i:06d}' for i in range(len(processed_df))]
        
        if 'price' not in processed_df.columns:
            processed_df['price'] = np.random.uniform(10, 1000, len(processed_df))
        
        # Features de performance
        self._add_performance_features(processed_df)
        
        # Target: lifecycle_stage
        self._create_lifecycle_target(processed_df)
        
        return processed_df
    
    def _add_performance_features(self, df):
        """Adiciona features de performance do produto"""
        n = len(df)
        
        if 'stock_quantity' not in df.columns:
            df['stock_quantity'] = np.random.randint(0, 500, n)
        
        if 'sales_last_30d' not in df.columns:
            # Correlacionar com preço (produtos mais baratos vendem mais)
            if 'price' in df.columns:
                price_factor = 1 / (df['price'] / df['price'].mean())
                df['sales_last_30d'] = (np.random.randint(0, 100, n) * price_factor).astype(int)
            else:
                df['sales_last_30d'] = np.random.randint(0, 200, n)
        
        if 'views_last_30d' not in df.columns:
            df['views_last_30d'] = df['sales_last_30d'] * np.random.randint(10, 50, n)
        
        if 'rating' not in df.columns:
            df['rating'] = np.random.uniform(1, 5, n)
        
        if 'num_reviews' not in df.columns:
            df['num_reviews'] = (df['sales_last_30d'] * np.random.uniform(0.1, 0.5, n)).astype(int)
        
        if 'days_since_launch' not in df.columns:
            df['days_since_launch'] = np.random.randint(1, 730, n)
        
        if 'discount_percentage' not in df.columns:
            df['discount_percentage'] = np.random.uniform(0, 50, n)
        
        if 'return_rate' not in df.columns:
            df['return_rate'] = np.random.uniform(0, 0.3, n)
        
        # Features derivadas
        df['conversion_rate'] = df['sales_last_30d'] / (df['views_last_30d'] + 1)
        df['revenue'] = df['price'] * df['sales_last_30d']
        df['revenue_per_view'] = df['revenue'] / (df['views_last_30d'] + 1)
        df['engagement_score'] = (df['num_reviews'] + df['rating'] * 10) / (df['days_since_launch'] + 1)
        
    def _create_lifecycle_target(self, df):
        """
        Cria target de ciclo de vida baseado em regras de negócio
        0: Descontinuar
        1: Manter
        2: Promover
        """
        
        # Normalizar métricas
        df['sales_score'] = (df['sales_last_30d'] - df['sales_last_30d'].min()) / (df['sales_last_30d'].max() - df['sales_last_30d'].min())
        df['rating_score'] = df['rating'] / 5.0
        df['conversion_score'] = (df['conversion_rate'] - df['conversion_rate'].min()) / (df['conversion_rate'].max() - df['conversion_rate'].min() + 0.001)
        
        # Score composto
        df['performance_score'] = (
            df['sales_score'] * 0.4 +
            df['rating_score'] * 0.3 +
            df['conversion_score'] * 0.3
        )
        
        # Regras de classificação
        conditions = [
            # Descontinuar: baixa performance E (alto estoque OU alta taxa de retorno)
            (df['performance_score'] < 0.3) & ((df['stock_quantity'] > df['stock_quantity'].quantile(0.7)) | (df['return_rate'] > 0.2)),
            
            # Promover: alta performance E boa avaliação
            (df['performance_score'] > 0.6) & (df['rating'] > 4.0),
        ]
        
        choices = [0, 2]
        df['lifecycle_stage'] = np.select(conditions, choices, default=1)
        
        print(f"\n📊 Distribuição do Target:")
        print(f"  Descontinuar: {(df['lifecycle_stage'] == 0).sum()} ({(df['lifecycle_stage'] == 0).sum()/len(df)*100:.1f}%)")
        print(f"  Manter: {(df['lifecycle_stage'] == 1).sum()} ({(df['lifecycle_stage'] == 1).sum()/len(df)*100:.1f}%)")
        print(f"  Promover: {(df['lifecycle_stage'] == 2).sum()} ({(df['lifecycle_stage'] == 2).sum()/len(df)*100:.1f}%)")