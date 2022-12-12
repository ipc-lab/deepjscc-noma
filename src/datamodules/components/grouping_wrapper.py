import math

import torch


class GroupingWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, csi, group_size, num_samples=-1, seed=1):
        super().__init__()
        self.dataset = dataset
        self.group_size = group_size

        self.csi = csi

        if num_samples == -1:
            self.indices = torch.combinations(
                torch.arange(0, len(dataset)), r=self.group_size, with_replacement=False
            )
            self.num_samples = self.indices.size(0)
        elif num_samples == 0:
            self.num_samples = math.ceil(float(len(self.dataset)) / self.group_size)
            num_new_elements = len(self.dataset)
            if len(self.dataset) % self.group_size != 0:
                num_new_elements += self.group_size - len(self.dataset) % self.group_size

            self.indices = torch.arange(0, num_new_elements) % len(self.dataset)
            self.indices = self.indices.reshape(-1, self.group_size)
        else:
            self.num_samples = num_samples
            # Generate random combinations of indices
            r = torch.randint(0, len(dataset), (3 * self.num_samples, self.group_size))
            self.indices = r.unique(sorted=False, dim=0)[: self.num_samples, :].long()

            if self.indices.size(0) < self.num_samples:
                self.num_samples = self.indices.size(0)

        shuffled_indices = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(seed)
        )

        self.indices = shuffled_indices[self.indices]

    def __getitem__(self, idx):
        """if self.with_replacement:"""

        indices = self.indices[idx]
        X = []
        y_single = torch.zeros(
            (self.group_size,),
            dtype=torch.long,
        )
        for i in range(self.group_size):
            x, cur_y = self.dataset.__getitem__(indices[i])
            X.append(x)
            if cur_y == 1:
                y_single[i] = 1

        return (torch.stack(X, dim=0), self.csi[idx])

    def __len__(self):

        return self.num_samples
