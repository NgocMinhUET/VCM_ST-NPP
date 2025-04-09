#!/usr/bin/env python3
"""
Script to generate a synthetic test video for training the STNPP model.
"""

import os
import cv2
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic test videos')
    parser.add_argument('--output_dir', type=str, default='test_data',
                        help='Directory to save the generated videos')
    parser.add_argument('--num_videos', type=int, default=5,
                        help='Number of videos to generate')
    parser.add_argument('--duration', type=int, default=5,
                        help='Duration of each video in seconds')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second')
    parser.add_argument('--width', type=int, default=224,
                        help='Video width in pixels')
    parser.add_argument('--height', type=int, default=224,
                        help='Video height in pixels')
    return parser.parse_args()

def generate_moving_box_video(filepath, duration=5, fps=30, width=224, height=224):
    """
    Generate a video with a moving box.
    
    Args:
        filepath: Output file path
        duration: Duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height
    """
    num_frames = duration * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    
    # Initial position and velocity of the box
    box_size = 30
    x, y = width // 4, height // 4
    vx, vy = 2, 3
    
    # Generate frames
    for i in range(num_frames):
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Update box position
        x += vx
        y += vy
        
        # Bounce if hitting the edges
        if x <= 0 or x + box_size >= width:
            vx = -vx
        if y <= 0 or y + box_size >= height:
            vy = -vy
        
        # Draw the box
        color = (0, 0, 255)  # Red
        cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), color, -1)
        
        # Add some text
        text = f"Frame: {i+1}/{num_frames}"
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add timestamp
        cv2.putText(frame, f"Time: {i/fps:.2f}s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Write the frame
        out.write(frame)
    
    out.release()
    print(f"Generated video saved to {filepath}")

def generate_moving_shapes_video(filepath, duration=5, fps=30, width=224, height=224):
    """
    Generate a video with multiple moving shapes.
    
    Args:
        filepath: Output file path
        duration: Duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height
    """
    num_frames = duration * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    
    # Create shapes
    class Shape:
        def __init__(self, x, y, vx, vy, color, shape_type, size):
            self.x = x
            self.y = y
            self.vx = vx
            self.vy = vy
            self.color = color
            self.shape_type = shape_type  # 'circle', 'rectangle', 'triangle'
            self.size = size
        
        def update(self):
            self.x += self.vx
            self.y += self.vy
            
            # Bounce if hitting the edges
            if self.x <= 0 or self.x + self.size >= width:
                self.vx = -self.vx
            if self.y <= 0 or self.y + self.size >= height:
                self.vy = -self.vy
        
        def draw(self, frame):
            if self.shape_type == 'circle':
                cv2.circle(frame, (self.x, self.y), self.size // 2, self.color, -1)
            elif self.shape_type == 'rectangle':
                cv2.rectangle(frame, (self.x, self.y), 
                             (self.x + self.size, self.y + self.size), self.color, -1)
            elif self.shape_type == 'triangle':
                points = np.array([
                    [self.x, self.y + self.size],
                    [self.x + self.size, self.y + self.size],
                    [self.x + self.size // 2, self.y]
                ], np.int32)
                cv2.fillPoly(frame, [points], self.color)
    
    # Create random shapes
    num_shapes = 5
    shapes = []
    
    for _ in range(num_shapes):
        shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        size = np.random.randint(20, 50)
        x = np.random.randint(0, width - size)
        y = np.random.randint(0, height - size)
        vx = np.random.randint(1, 5) * (1 if np.random.random() > 0.5 else -1)
        vy = np.random.randint(1, 5) * (1 if np.random.random() > 0.5 else -1)
        
        shapes.append(Shape(x, y, vx, vy, color, shape_type, size))
    
    # Generate frames
    for i in range(num_frames):
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Update and draw shapes
        for shape in shapes:
            shape.update()
            shape.draw(frame)
        
        # Add some text
        text = f"Frame: {i+1}/{num_frames}"
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add timestamp
        cv2.putText(frame, f"Time: {i/fps:.2f}s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Write the frame
        out.write(frame)
    
    out.release()
    print(f"Generated video saved to {filepath}")

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate videos
    for i in range(args.num_videos):
        # Alternate between video types
        if i % 2 == 0:
            filepath = os.path.join(args.output_dir, f"box_video_{i+1}.mp4")
            generate_moving_box_video(filepath, args.duration, args.fps, args.width, args.height)
        else:
            filepath = os.path.join(args.output_dir, f"shapes_video_{i+1}.mp4")
            generate_moving_shapes_video(filepath, args.duration, args.fps, args.width, args.height)
    
    print(f"Generated {args.num_videos} videos in {args.output_dir}.")

if __name__ == "__main__":
    main() 