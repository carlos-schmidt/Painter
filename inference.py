import torch
import torchvision
from torchvision.transforms.functional import pil_to_tensor

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

        grid = torch.zeros(size=(top.shape[0], top.shape[1]*2, top.shape[2]))

        grid[:, : top.shape[1], : top.shape[2]] = top
        grid[:, bottom.shape[1] : , : bottom.shape[2]] = bottom

        return grid

    @torch.no_grad()
    def inference(self, img_batch):
        """Inference on batch of images

        Args:
            img_batch (list): List of list of PIL images
        """
        batch_size = len(img_batch)
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
        #pred = torch.nn.functional.interpolate(            pred.permute(0, 3, 1, 2), original_sizes, mode="bicubic"        ).permute(0, 2, 3, 1)

        return pred


if __name__ == "__main__":
    from PIL import Image
    import requests
    from io import BytesIO
    import numpy as np
    painter = Painter(model_path="./painter_vit_large.pth")

    def url_to_pil(url) -> Image.Image:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img

    # These are from the visual_prompting repo
    source = "https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2022/06/14/ML-8362-image001-300.jpg"
    target = "https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2022/06/14/ML-8362-image003-300.png"
    new_source = "https://static.scientificamerican.com/sciam/cache/file/1E3A3E62-B3CA-434A-8C3B3ED0C982FB69_source.jpg?w=590&h=800&C8DB8C57-989B-4118-AE27EF1191E878A5"

    imgs = painter.inference(
        [
            [
                url_to_pil(source).convert("RGB"),
                url_to_pil(target).convert("RGB"),
                url_to_pil(new_source).convert("RGB"),
            ],
            [
                url_to_pil(source).convert("RGB"),
                url_to_pil(target).convert("RGB"),
                url_to_pil(new_source).convert("RGB"),
            ],
        ]
    )

    print(imgs.shape, imgs[0].shape)
    out = imgs[0] * 255.
    out = out.detach().cpu().numpy().astype(np.uint8)
    
    Image.fromarray(out).save("test.png")
