def resource_allocation(model,Number_of_gpus = None):

    # Add features to use number of GPUS

    device = "cpu"

    if torch.cuda.is_available():

        device = "cuda:0"
            if torch.cuda.device_count() > 1:
                if Number_of_gpus == None:
                    model = nn.DataParallel(model)
                else:
                    model = nn.DataParallel(model, device_ids = [i for i in range(Number_of_gpus)])

    model.to(device)
    return model

