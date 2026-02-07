def freeze_backbone(model):
    for name, param in model.named_parameters():
        if any(key in name for key in ["classifier", "head", "fc","logits"]):
            param.requires_grad = True
        else:
            param.requires_grad = False


def freeze_backbone_partial(model, model_name):

    # freeze everything except head
    freeze_backbone(model)

    layers_to_unfreeze = []
    
    if "convnext" in model_name:
        # ConvNeXt: 'features.7' last bloc
        layers_to_unfreeze.append("features.7") 
    elif "resnet" in model_name:
        # ResNet: 'layer4' last layer
        layers_to_unfreeze.append("layer3")
        layers_to_unfreeze.append("layer4")
    
    elif "vggface" in model_name:
        # Correspond à ta logique Colab : 
        # mixed_7a, block8, last_linear, last_bn
        layers_to_unfreeze.append("mixed_7a")
        layers_to_unfreeze.append("block8")
        layers_to_unfreeze.append("last_linear")
        layers_to_unfreeze.append("last_bn")
    
    print(f"Partial Freeze active for {model_name}: {layers_to_unfreeze}")
    
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_unfreeze):
            param.requires_grad = True

def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True


def get_param_groups(model, backbone_lr, head_lr):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(key in name for key in ["classifier", "head", "fc"]):
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]
