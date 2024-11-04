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

    def images_to_grid(self, left, right):
        """Image tensors of shape c,h,w"""
        assert left.shape == right.shape

        grid = torch.zeros(size=(left.shape[0], left.shape[1], left.shape[2] * 2))

        grid[:, : left.shape[1], : left.shape[2]] = left
        grid[:, : right.shape[1], right.shape[2] + 1 :] = right

        return grid

    @torch.no_grad()
    def inference(self, img_batch):
        """Inference on batch of images

        Args:
            img_batch (list): List of list of PIL images
        """
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

        """ TODO maybe we don't need this since Painter.forward() does it
        # black out bottom right image
        num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
        bool_masked_pos = torch.zeros(num_patches)
        bool_masked_pos[num_patches // 2 :] = 1
        # Do this for all in batch
        bool_masked_pos = bool_masked_pos.unsqueeze(dim=0).repeat(batch_size, 1, 1, 1)
        """

        _, pred, _ = self.model(prompt_grids, ctx_grids)

        pred = self.model.unpatchify(pred)
        pred = torch.einsum("bchw->bhwc", pred).detach().cpu()

        # Get real targets
        pred = pred[:, pred.shape[1] // 2 :, :, :]

        # pred = clamp((pred * imagenet_std + imagenet_mean), 0.0, 1.0) ?

        # Resize to original
        pred = torch.nn.functional.interpolate(
            pred.permute(0, 3, 1, 2), original_sizes, mode="bicubic"
        ).permute(0, 2, 3, 1)

        return pred


if __name__ == "__main__":
    from PIL import Image

    painter = Painter(chkpt_dir="./painter_vit_large.pth", device="cpu")
    ctx_in = Image.open("./image-1.png")
    ctx_tgt = Image.open("./image-2.jpg")
    prompt = Image.open("./image-3.jpg")

    out = painter.inference([ctx_in, ctx_tgt, prompt])
    Image.fromarray(out).show()
