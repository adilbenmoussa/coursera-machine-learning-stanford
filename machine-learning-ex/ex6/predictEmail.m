function prediction = predictEmail(email,  model)

    file_contents = readFile(email);
    word_indices = processEmail(file_contents);
    features = emailFeatures(word_indices);
    prediction = svmPredict(model, features);
