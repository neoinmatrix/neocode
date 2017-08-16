 cmd = ['-v ', num2str(5),' -c ',num2str(cv(1)),' -g ',num2str(cv(2))];
fitness=svmtrain(train_label,train_sample,cmd); % SVM模型训练
