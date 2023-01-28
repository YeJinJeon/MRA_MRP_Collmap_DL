from training_model_wrapper.training_model_wrapper import TrainingModelWrapper
from torch.autograd import Variable
import torch

class TrainingModelWrapperFor5d(TrainingModelWrapper):

    def calculate_train_loss(self):
        training_loss, reg_loss, ord_reg_loss = 0
        for batch_idx, (inputs, *labels, label_weight, mask, file) in enumerate(self.training_generator):
            # print(file)
            inputs = Variable(inputs.to(self.device))
            label_weight = Variable(label_weight.to(self.device))
            mask = Variable(mask.to(self.device))
            labels = list(labels)
            new_labels_list = []
            for idx, label in enumerate(labels):
                # label[0] is normal label
                # label[x] with x > 0 is ordinal label
                if idx == 0:
                    new_labels_list.append(Variable(label.to(self.device)))
                elif idx > 0:
                    label[mask.repeat(1, 5, 1, 1, 1) == 0] = -1
                    new_label = Variable(label.to(self.device))
                    new_labels_list.append(new_label)
            predicts = self.network_architecture(inputs)
            loss = 0
            if "UncertaintyLoss" in str(type(self.lost_list[0])) or "TestLoss" in str(type(self.lost_list[0])):
                loss += self.lost_list[0](predicts, new_labels_list, label_weight, mask)
            elif len(self.lost_list) == 1:
                loss += self.lost_list[0](predicts, new_labels_list[0], label_weight, mask)
            else:
                loss_list = []
                for idx, (criteria, criteria_weight) in enumerate(zip(self.lost_list, self.loss_weights)):
                    predict = predicts[idx] if idx < len(predicts) else predicts[len(predicts) - 1]
                    if idx > 0:
                        loss = loss.to(criteria.device)
                    loss_element = criteria(predict, new_labels_list[idx], label_weight, mask)
                    loss_list.append(loss_element)
                    loss += criteria_weight * loss_element
            self.optimizer.zero_grad()
            loss.to(self.device)
            loss.backward()
            self.optimizer.step()
            training_loss += float(loss.item())
            reg_loss += float(loss_list[0].item())
            ord_reg_loss += float(loss_list[1].item())
        return training_loss, reg_loss, ord_reg_loss

    def calculate_validate_loss(self):
        validation_loss, reg_loss, ord_reg_loss = 0
        self.network_architecture.eval()
        with torch.no_grad():
            for index, (inputs, labels, label_weight, mask, file) in enumerate(self.validation_generator):
                # print(file)
                inputs = Variable(inputs.to(self.device))
                labels = list(labels)
                new_labels_list = []
                for idx, label in enumerate(labels):
                    # label[0] is normal label
                    # label[x] with x > 0 is ordinal label
                    if idx == 0:
                        new_labels_list.append(Variable(label.to(self.device)))
                    elif idx > 0:
                        label[mask.repeat(1, 5, 1, 1, 1) == 0] = -1
                        new_label = Variable(label.to(self.device))
                        new_labels_list.append(new_label)
                label_weight = Variable(label_weight.to(self.device))
                mask = Variable(mask.to(self.device))
                predicts = self.network_architecture(inputs)
                loss = 0
                if "UncertaintyLoss" in str(type(self.val_loss_list[0])) or "TestLoss" in str(type(self.val_loss_list[0])):
                    loss += self.val_loss_list[0](predicts, new_labels_list, label_weight, mask)
                elif len(self.val_loss_list) == 1:
                    if type(predicts) is tuple:
                        loss += self.val_loss_list[0](predicts[0], new_labels_list[0], label_weight, mask)
                    else:
                        loss += self.val_loss_list[0](predicts, new_labels_list[0], label_weight, mask)
                else:
                    loss_list = []
                    for idx, (criteria, criteria_weight) in enumerate(zip(self.val_loss_list, self.val_loss_weights)):
                        predict = predicts[idx] if idx < len(predicts) else predicts[len(predicts) - 1]
                        loss_element = criteria(predict, new_labels_list[idx], label_weight, mask)
                        loss_list.append(loss_element)
                        loss += criteria_weight * loss_element
                validation_loss += loss
                reg_loss += float(loss_list[0])
                ord_reg_loss += float(loss_list[1])
        if "UncertaintyLoss" in str(type(self.val_loss_list[0])):
            for idx, p in enumerate(self.val_loss_list[0].params):
                print(f"{idx}: {p}")
        return validation_loss, reg_loss, ord_reg_loss
