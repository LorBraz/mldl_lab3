import models

model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_acc = 0

# Run the training process for {num_epochs} epochs
num_epochs = 20
for epoch in range(1, num_epochs + 1):
    train(epoch, model, train_loader, criterion, optimizer)

