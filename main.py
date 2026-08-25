from src.components.data_ingestion import load_raw_data


def main():
    data_path = "data/student-mat.csv"

    df = load_raw_data(data_path)

    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()
