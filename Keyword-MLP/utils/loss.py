import torch
from torch import nn

class LabelSmoothingLoss(nn.Module):
    """Cross Entropy with Label Smoothing.
    
    Attributes:
        num_classes (int): Number of target classes.
        smoothing (float, optional): Smoothing fraction constant, in the range (0.0, 1.0). Defaults to 0.1.
        dim (int, optional): Dimension across which to apply loss. Defaults to -1.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1, dim: int = -1):
        """Initializer for LabelSmoothingLoss.

        Args:
            num_classes (int): Number of target classes.
            smoothing (float, optional): Smoothing fraction constant, in the range (0.0, 1.0). Defaults to 0.1.
            dim (int, optional): Dimension across which to apply loss. Defaults to -1.
        """
        super().__init__()
        
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = num_classes
        self.dim = dim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): Model predictions, of shape (batch_size, num_classes) or a tuple where the first element is the predictions.
            target (torch.Tensor): Target tensor of shape (batch_size).

        Returns:
            torch.Tensor: Loss.
        """
        assert 0 <= self.smoothing < 1
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class MultiTaskLoss(nn.Module):
    """
    A simple multi-task loss that sums the cross-entropy losses of two heads:
      1) Keyword classifier (e.g. 35 classes)
      2) Language classifier (e.g. 3 classes)

    The default weighting is w_kw=1.0, w_lang=1.0, but you can adjust as needed.
    """
    def __init__(self, w_kw: float = 1.0, w_lang: float = 1.0):
        """
        Args:
            w_kw (float): Weight for keyword classification loss.
            w_lang (float): Weight for language classification loss.
        """
        super().__init__()
        self.w_kw = w_kw
        self.w_lang = w_lang
        # We'll just use standard cross-entropy for both tasks here.
        self.ce_kw = nn.CrossEntropyLoss()
        self.ce_lang = nn.CrossEntropyLoss()
        # Store the most recent individual losses
        self.last_loss_kw = None
        self.last_loss_lang = None

    def forward(
        self,
        logits_kw: torch.Tensor,
        logits_lang: torch.Tensor,
        target_kw: torch.Tensor,
        target_lang: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits_kw (torch.Tensor): [B, num_kw_classes] (e.g., 35)
            logits_lang (torch.Tensor): [B, num_lang_classes] (e.g., 3)
            target_kw (torch.Tensor): Keyword labels, shape [B]
            target_lang (torch.Tensor): Language labels, shape [B]

        Returns:
            torch.Tensor: Combined weighted loss
        """
        self.last_loss_kw = self.ce_kw(logits_kw, target_kw)
        self.last_loss_lang = self.ce_lang(logits_lang, target_lang)
        return self.w_kw * self.last_loss_kw + self.w_lang * self.last_loss_lang

    def get_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the most recent individual losses.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of (keyword_loss, language_loss)
        """
        return self.last_loss_kw, self.last_loss_lang