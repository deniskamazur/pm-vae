class WeirdEncoder(nn.Module):
    def __init__(self, n_approx=5):
        super(self.__class__, self).__init__()
        
        self.n_approx = n_approx
        
        self.layers = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU()
        )
        
        
        self.mu = nn.Linear(400, 20 * n_approx)
        self.sigma = nn.Linear(400, 20 * n_approx)
        self.decision = nn.Linear(400, 20 * n_approx)
        
    def encode(self, x):
        x = self.layers(x)
        dec = gumbel_softmax(self.decision(x).reshape(-1, self.n_approx), temperature=0.1).reshape(-1, 20, self.n_approx)
        mu  = self.mu(x).reshape(-1, 20, self.n_approx)
        sma = self.sigma(x).reshape(-1, 20, self.n_approx)
        
        return mu * dec, sma * dec
    
