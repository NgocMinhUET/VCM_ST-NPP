import os
import datetime
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

def train(args):
    """Main training function."""
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    stnpp_output_dir = os.path.join(args.output_dir, args.stnpp_dir)
    qal_output_dir = os.path.join(args.output_dir, args.qal_dir)
    os.makedirs(stnpp_output_dir, exist_ok=True)
    os.makedirs(qal_output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set up TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'stnpp_training_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models
    print("Initializing ST-NPP model...")
    try:
        stnpp_model = STNPP(
            input_channels=3,
            output_channels=args.output_channels,
            spatial_backbone=args.stnpp_backbone,
            temporal_model=args.temporal_model,
            fusion_type=args.fusion_type,
            pretrained=True
        ).to(device)
    except Exception as e:
        print(f"Error initializing ST-NPP model: {e}")
        raise
    
    # Initialize QAL model based on type
    print(f"Initializing QAL model with type: {args.qal_type}")
    try:
        if args.qal_type == 'standard':
            qal_model = QAL(
                feature_channels=args.output_channels,
                hidden_dim=64
            ).to(device)
        elif args.qal_type == 'conditional':
            qal_model = ConditionalQAL(
                feature_channels=args.output_channels,
                hidden_dim=64,
                kernel_size=3,
                temporal_kernel_size=3
            ).to(device)
        elif args.qal_type == 'pixelwise':
            qal_model = PixelwiseQAL(
                feature_channels=args.output_channels,
                hidden_dim=64
            ).to(device)
        else:
            raise ValueError(f"Unknown QAL type: {args.qal_type}")
    except Exception as e:
        print(f"Error initializing QAL model: {e}")
        raise
    
    # Set up optimizers
    stnpp_optimizer = optim.Adam(stnpp_model.parameters(), lr=args.lr)
    qal_optimizer = optim.Adam(qal_model.parameters(), lr=args.lr) 