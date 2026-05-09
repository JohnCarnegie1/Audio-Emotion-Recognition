from src import RawAudioDataset, create_dataloaders, BiLSTM, train, get_data_transforms


if __name__ == "__main__":
    # --- Load data ---
    transformed = get_data_transforms()
    dataset = RawAudioDataset(
        spectrogram_folder="wav_corpus", 
        label_file_path="label.txt", 
        label_map={'fear': 0, 'disgust': 1, 'happiness': 2, 'anger': 3, 'sadness': 4},
        transform=transformed
    )

    train_loader, val_loader, test_loader = create_dataloaders(dataset, batch_size=4)

    # --- Build model ---
    model = BiLSTM()

    # --- Train ---
    trained_model, train_losses, val_losses, val_accuracies = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=70,           
        lr=1e-4,
        device="cuda"        # ensure GPU usage
    )

    print("Training complete!")
