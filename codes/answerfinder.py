loss_history= []
model= customClass(400)
criterion= torch.nn.BCELoss(weight=torch.tensor([1, 1, 1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01]))
optimizer= torch.optim.Adam(model.parameters(), lr= 0.0001)
for tata in range(10):
    for j, data in enumerate(dataset):
        data.shuffle()
        pred= []; target= []
        quests= data.quests; docs= data.docs
        for i, query in enumerate(quests):
            pred.append([]); target.append([])
            q_vector= query.vector
            for context in docs:
                c_vector= context.vector
                y= torch.tensor([float(query.isRelated(context))], dtype= torch.float)
                y_pred= model(q_vector, c_vector)
                pred[i].append(y_pred); target[i].append(y)
    
         
        pred= torch.stack(list(map(torch.stack, pred)))
        target= torch.stack(list(map(torch.stack, target))); target= target.view([10, 10])
        loss= criterion(pred, target)
        if j+1 % 20 == 0:
            loss_history.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()