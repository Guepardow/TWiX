import torch


class BidirectionalContrastiveLoss:

    def __init__(self, tau=0.1, B=1024, device='cuda'):
        self.tau = tau
        self.B = B  # number of contrasts
        self.device = device

    def __call__(self, output, target):
        """
        :param output: a 2D tensor with values between -1 and 1 indicating the similarity between objects
        :param target: a 2D tensor with values in {-1, 0, 1}
        """
        
        total_loss = torch.tensor(0, dtype=torch.float64, requires_grad=True, device=self.device)

        # Get positive pairs
        posi, posj = torch.where(target == 1)       # P and P

        if len(posi) == 0:
            return total_loss

        # Create a mask for target where values are 0
        Y0 = (target == 0).clone().detach()              # M x N

        # Select the output values for positive pairs
        output_pos = output[posi, posj]  # P

        # Create a mask for the rows in output corresponding to Y0
        neg_indices_forward = Y0[posi]       # P x N
        neg_indices_backward = Y0[:, posj]   # M x P

        # Gather the outputs for valid negative indices
        neg_outputs_forward = output[posi, :] - output_pos[:, None]  # P x N
        neg_outputs_backward = output[:, posj] - output_pos[None, :] # M x P

        # Select negative values 
        neg_outputs_forward[~neg_indices_forward] = float('nan')     # P x N
        neg_outputs_backward[~neg_indices_backward] = float('nan')   # M x P

        # Calculate the exponentials
        neg_values_forward_exp = torch.exp(neg_outputs_forward / self.tau)    # P x N
        neg_values_backward_exp = torch.exp(neg_outputs_backward / self.tau)  # M x P

        # Calculate losses for valid negative values
        loss_forward = torch.log1p(self.B * neg_values_forward_exp.nanmean(dim=1))    # P 
        loss_backward = torch.log1p(self.B * neg_values_backward_exp.nanmean(dim=0))  # P 

        # Number of positive pairs linked to at least one negative pair
        n_pos_forward = torch.count_nonzero(~torch.isnan(loss_forward))
        n_pos_backward = torch.count_nonzero(~torch.isnan(loss_backward))

        # Forward loss
        if n_pos_forward != 0:
            total_loss = total_loss + loss_forward.nansum() / n_pos_forward

        # Backward loss
        if n_pos_backward != 0:
            total_loss = total_loss + loss_backward.nansum() / n_pos_backward

        return total_loss
    

