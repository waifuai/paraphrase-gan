"""
Data Analysis Plugin

This plugin provides comprehensive data analysis and processing capabilities for the application.
It extends the core functionality with tools for statistical analysis, data visualization,
file format conversion, and data cleaning/preprocessing operations.

The plugin includes methods for loading and saving data in multiple formats (CSV, TSV, JSON, Excel, Parquet),
performing statistical calculations, filtering data based on complex conditions, converting
data types, and aggregating data by grouping columns.

This plugin provides data analysis and processing capabilities including:
- Statistical analysis
- Data visualization
- File format conversion
- Data cleaning and preprocessing
"""

import pandas as pd
import numpy as np
import json
import csv
import os
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import statistics
from pathlib import Path


class DataAnalysis:
    """Data analysis and processing tools."""

    def __init__(self):
        self.supported_formats = ['csv', 'json', 'tsv', 'xlsx', 'parquet']

    def load_data(self, file_path: str, file_type: Optional[str] = None) -> Dict[str, Union[bool, str, Any]]:
        """Load data from various file formats."""
        try:
            path = Path(file_path)

            if not path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            # Auto-detect file type if not specified
            if file_type is None:
                file_type = path.suffix.lower().lstrip('.')

            if file_type not in self.supported_formats:
                return {"success": False, "error": f"Unsupported format: {file_type}"}

            data = None
            if file_type == 'csv':
                data = pd.read_csv(file_path)
            elif file_type == 'tsv':
                data = pd.read_csv(file_path, sep='\t')
            elif file_type == 'json':
                data = pd.read_json(file_path)
            elif file_type == 'xlsx':
                data = pd.read_excel(file_path)
            elif file_type == 'parquet':
                data = pd.read_parquet(file_path)

            return {
                "success": True,
                "data": data,
                "shape": data.shape,
                "columns": list(data.columns),
                "file_type": file_type
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_data(self, data: pd.DataFrame, file_path: str, file_type: Optional[str] = None) -> Dict[str, Union[bool, str]]:
        """Save data to various file formats."""
        try:
            path = Path(file_path)

            # Auto-detect file type if not specified
            if file_type is None:
                file_type = path.suffix.lower().lstrip('.')

            if file_type not in self.supported_formats:
                return {"success": False, "error": f"Unsupported format: {file_type}"}

            # Create directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            if file_type == 'csv':
                data.to_csv(file_path, index=False)
            elif file_type == 'tsv':
                data.to_csv(file_path, sep='\t', index=False)
            elif file_type == 'json':
                data.to_json(file_path, orient='records', indent=2)
            elif file_type == 'xlsx':
                data.to_excel(file_path, index=False)
            elif file_type == 'parquet':
                data.to_parquet(file_path, index=False)

            return {"success": True, "message": f"Data saved to {file_path}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive information about a dataset."""
        try:
            info = {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": data.dtypes.to_dict(),
                "memory_usage": data.memory_usage(deep=True).sum(),
                "null_counts": data.isnull().sum().to_dict(),
                "duplicate_count": data.duplicated().sum()
            }

            # Basic statistics for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                info["numeric_stats"] = data[numeric_cols].describe().to_dict()

            # Value counts for categorical columns (top 10)
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            cat_info = {}
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                value_counts = data[col].value_counts().head(10).to_dict()
                cat_info[col] = value_counts
            info["categorical_info"] = cat_info

            return {"success": True, "info": info}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def clean_data(self, data: pd.DataFrame, operations: List[str]) -> Dict[str, Union[bool, str, pd.DataFrame]]:
        """Clean data with various operations."""
        try:
            cleaned_data = data.copy()

            for operation in operations:
                if operation == 'drop_nulls':
                    cleaned_data = cleaned_data.dropna()
                elif operation == 'drop_duplicates':
                    cleaned_data = cleaned_data.drop_duplicates()
                elif operation == 'fill_nulls_mean':
                    numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
                    cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(cleaned_data[numeric_cols].mean())
                elif operation == 'fill_nulls_median':
                    numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
                    cleaned_data[numeric_cols] = cleaned_data[numeric_cols].fillna(cleaned_data[numeric_cols].median())
                elif operation == 'fill_nulls_mode':
                    for col in cleaned_data.columns:
                        if cleaned_data[col].isnull().any():
                            mode_val = cleaned_data[col].mode()
                            if len(mode_val) > 0:
                                cleaned_data[col] = cleaned_data[col].fillna(mode_val[0])
                elif operation == 'strip_whitespace':
                    string_cols = cleaned_data.select_dtypes(include=['object']).columns
                    for col in string_cols:
                        cleaned_data[col] = cleaned_data[col].astype(str).str.strip()
                elif operation == 'lowercase_strings':
                    string_cols = cleaned_data.select_dtypes(include=['object']).columns
                    for col in string_cols:
                        cleaned_data[col] = cleaned_data[col].str.lower()

            return {
                "success": True,
                "data": cleaned_data,
                "original_shape": data.shape,
                "cleaned_shape": cleaned_data.shape,
                "operations_applied": operations
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def calculate_statistics(self, data: pd.DataFrame, column: str) -> Dict[str, Union[bool, str, float, int]]:
        """Calculate comprehensive statistics for a column."""
        try:
            if column not in data.columns:
                return {"success": False, "error": f"Column '{column}' not found"}

            series = data[column].dropna()

            if len(series) == 0:
                return {"success": False, "error": f"Column '{column}' has no valid values"}

            stats = {
                "count": len(series),
                "null_count": data[column].isnull().sum(),
                "unique_count": series.nunique(),
                "most_common": series.mode().iloc[0] if len(series.mode()) > 0 else None,
                "most_common_count": series.value_counts().iloc[0] if len(series.value_counts()) > 0 else 0
            }

            # Numeric statistics
            if pd.api.types.is_numeric_dtype(series):
                stats.update({
                    "mean": series.mean(),
                    "median": series.median(),
                    "mode": series.mode().iloc[0] if len(series.mode()) > 0 else None,
                    "std": series.std(),
                    "var": series.var(),
                    "min": series.min(),
                    "max": series.max(),
                    "q25": series.quantile(0.25),
                    "q75": series.quantile(0.75),
                    "iqr": series.quantile(0.75) - series.quantile(0.25)
                })

                # Additional statistics
                try:
                    stats["skewness"] = series.skew()
                    stats["kurtosis"] = series.kurtosis()
                except:
                    pass

            # String/categorical statistics
            elif pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series):
                stats.update({
                    "avg_length": series.astype(str).str.len().mean(),
                    "max_length": series.astype(str).str.len().max(),
                    "min_length": series.astype(str).str.len().min()
                })

            return {"success": True, "column": column, "statistics": stats}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def filter_data(self, data: pd.DataFrame, conditions: List[Dict[str, Any]]) -> Dict[str, Union[bool, str, pd.DataFrame]]:
        """Filter data based on multiple conditions."""
        try:
            filtered_data = data.copy()

            for condition in conditions:
                column = condition.get('column')
                operator = condition.get('operator', '==')
                value = condition.get('value')

                if not column or column not in filtered_data.columns:
                    continue

                if operator == '==':
                    filtered_data = filtered_data[filtered_data[column] == value]
                elif operator == '!=':
                    filtered_data = filtered_data[filtered_data[column] != value]
                elif operator == '>':
                    filtered_data = filtered_data[filtered_data[column] > value]
                elif operator == '<':
                    filtered_data = filtered_data[filtered_data[column] < value]
                elif operator == '>=':
                    filtered_data = filtered_data[filtered_data[column] >= value]
                elif operator == '<=':
                    filtered_data = filtered_data[filtered_data[column] <= value]
                elif operator == 'contains':
                    filtered_data = filtered_data[filtered_data[column].astype(str).str.contains(str(value), case=False)]
                elif operator == 'startswith':
                    filtered_data = filtered_data[filtered_data[column].astype(str).str.startswith(str(value))]
                elif operator == 'endswith':
                    filtered_data = filtered_data[filtered_data[column].astype(str).str.endswith(str(value))]

            return {
                "success": True,
                "data": filtered_data,
                "original_count": len(data),
                "filtered_count": len(filtered_data),
                "conditions_applied": conditions
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def convert_data_types(self, data: pd.DataFrame, conversions: Dict[str, str]) -> Dict[str, Union[bool, str, pd.DataFrame]]:
        """Convert data types for specified columns."""
        try:
            converted_data = data.copy()

            type_mapping = {
                'int': 'int64',
                'float': 'float64',
                'str': 'string',
                'bool': 'boolean',
                'datetime': 'datetime64[ns]'
            }

            for column, target_type in conversions.items():
                if column not in converted_data.columns:
                    continue

                pandas_type = type_mapping.get(target_type.lower(), target_type)

                try:
                    if target_type.lower() == 'datetime':
                        converted_data[column] = pd.to_datetime(converted_data[column])
                    else:
                        converted_data[column] = converted_data[column].astype(pandas_type)
                except Exception as e:
                    return {"success": False, "error": f"Failed to convert column '{column}' to {target_type}: {str(e)}"}

            return {
                "success": True,
                "data": converted_data,
                "conversions_applied": conversions,
                "new_dtypes": converted_data.dtypes.to_dict()
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def aggregate_data(self, data: pd.DataFrame, group_by: List[str], aggregations: Dict[str, List[str]]) -> Dict[str, Union[bool, str, pd.DataFrame]]:
        """Aggregate data by grouping columns."""
        try:
            # Validate that group_by columns exist
            for col in group_by:
                if col not in data.columns:
                    return {"success": False, "error": f"Group by column '{col}' not found"}

            # Create aggregation dictionary
            agg_dict = {}
            for col, operations in aggregations.items():
                if col not in data.columns:
                    continue

                if col in group_by:
                    continue  # Skip group by columns

                agg_dict[col] = operations

            # Perform aggregation
            aggregated = data.groupby(group_by).agg(agg_dict).reset_index()

            # Flatten column names if needed
            if isinstance(aggregated.columns, pd.MultiIndex):
                aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]

            return {
                "success": True,
                "data": aggregated,
                "group_by": group_by,
                "aggregations": aggregations,
                "result_shape": aggregated.shape
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


# Plugin registration
def register_plugin():
    """Register the data analysis plugin."""
    return {
        "name": "data_analysis",
        "description": "Data analysis and processing tools",
        "version": "1.0.0",
        "class": DataAnalysis,
        "methods": [
            "load_data",
            "save_data",
            "get_data_info",
            "clean_data",
            "calculate_statistics",
            "filter_data",
            "convert_data_types",
            "aggregate_data"
        ]
    }