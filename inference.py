import torch
import torchvision

from models_painter import (
    painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1 as painter_config,
)

IMAGE_SIZE = 448
PATCH_SIZE = 16
image_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        torchvision.transforms.ToTensor(),
    ]
)


class Painter:
    def __init__(self, model_path="./painter_vit_large.pth", device="cuda") -> None:
        self.model = painter_config()
        checkpoint = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(device)
        self.device = device

    def images_to_grid(self, top, bottom):
        """Image tensors of shape c,h**22,w"""
        assert top.shape == bottom.shape

        grid = torch.zeros(size=(top.shape[0], top.shape[1] * 2, top.shape[2]))

        grid[:, : top.shape[1], : top.shape[2]] = top
        grid[:, bottom.shape[1] :, : bottom.shape[2]] = bottom

        return grid

    @torch.no_grad()
    def inference(self, img_batch):
        """Inference on batch of images

        Args:
            img_batch (list): List of list of PIL images
        """
        batch_size = len(img_batch)
        if type(img_batch[0]) is not torch.Tensor:
            img_batch = [[image_transform(image) for image in b] for b in img_batch]

        ctx_grids = torch.stack(
            [self.images_to_grid(b[-3], b[-2]) for b in img_batch]
        ).to(self.device)
        # Right image can be anything because of masking
        prompt_grids = torch.stack(
            [self.images_to_grid(b[-1], b[-1]) for b in img_batch]
        ).to(self.device)

        original_sizes = torch.stack(
            [torch.stack([im for im in b]) for b in img_batch]
        ).shape

        # black out bottom right image
        num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
        bool_masked_pos = torch.zeros(num_patches)
        bool_masked_pos[num_patches // 2 :] = 1
        # Do this for all in batch
        bool_masked_pos = bool_masked_pos.unsqueeze(dim=0).repeat(batch_size, 1, 1, 1)

        _, pred, _ = self.model(prompt_grids, ctx_grids, bool_masked_pos)

        pred = self.model.unpatchify(pred)
        pred = torch.einsum("bchw->bhwc", pred).detach().cpu()

        # Get real targets
        pred = pred[:, pred.shape[1] // 2 :, :, :]

        # pred = clamp((pred * imagenet_std + imagenet_mean), 0.0, 1.0) ?

        # Resize to original
        # pred = torch.nn.functional.interpolate(            pred.permute(0, 3, 1, 2), original_sizes, mode="bicubic"        ).permute(0, 2, 3, 1)

        return pred


if __name__ == "__main__":
    model = Painter(model_path="./painter_vit_large.pth")

    from dataset import InteractiveDataset
    from torchvision.transforms.functional import to_pil_image, resize, to_tensor
    from torchvision.transforms import Compose, Resize, InterpolationMode, ToTensor
    from torch.utils.data import DataLoader
    from os import makedirs

    resize = Compose(
        [
            Resize((111, 111), interpolation=InterpolationMode.BILINEAR),
            ToTensor(),
        ]
    )

    dataset = InteractiveDataset(
        "/home/carlos/VOC2012",
        "InteractionsMerged",
        "SegmentationSingleObjects",
        transform=resize,
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    for i, batch in enumerate(dataloader):
        model_input = []
        for image, gt in zip(*batch.values()):
            model_input.append(image)
            model_input.append(gt)

        # remove last gt
        outputs = model.inference([torch.stack(model_input[:-1])])

        # bottom right corner
        model_input.append(outputs[0][:, 113:, 113:])
        model_input.append(resize(to_pil_image(outputs[0])))
        catted = torch.cat(model_input, dim=2)

        makedirs("./interactive_demo/", exist_ok=True)
        to_pil_image(catted).save(f"./interactive_demo/{i}.png")
