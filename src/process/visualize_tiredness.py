"""
Visualization script for tiredness index results
Creates boxplots showing tiredness scores across subjects and sessions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path


def extract_subject_session_from_path(filepath):
    """
    Extract subject and session from file path like:
    /work/long_lab/Mojtaba/data/DROZY/data/original/psg/1-2.edf
    Returns: subject='1', session='2'
    """
    filename = Path(filepath).stem
    
    # Try hyphen separator (1-2)
    if '-' in filename:
        parts = filename.split('-', 1)
        subject = parts[0].strip()
        session = parts[1].strip()
    # Try underscore separator
    elif '_' in filename:
        parts = filename.split('_')
        if len(parts) >= 2:
            subject = parts[0].strip()
            session = parts[1].strip()
        else:
            subject = filename
            session = 'unknown'
    else:
        subject = filename
        session = 'unknown'
    
    return subject, session


def load_and_prepare_data(csv_file):
    """Load CSV and prepare data for plotting"""
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract subject and session from file_path if not already in columns
    if 'subject_id' not in df.columns or 'session_id' not in df.columns:
        print("Extracting subject_id and session_id from file_path...")
        df[['subject_id', 'session_id']] = df['file_path'].apply(
            lambda x: pd.Series(extract_subject_session_from_path(x))
        )
    
    # Convert to numeric
    df['subject_id'] = pd.to_numeric(df['subject_id'], errors='coerce')
    df['session_id'] = pd.to_numeric(df['session_id'], errors='coerce')
    
    # Create combined label
    df['subject_session'] = df['subject_id'].astype(str) + '-' + df['session_id'].astype(str)
    
    print(f"\nUnique subjects: {df['subject_id'].nunique()}")
    print(f"Unique sessions per subject: {df.groupby('subject_id')['session_id'].nunique().to_dict()}")
    
    return df


def plot_boxplot_by_subject(df, output_file=None, figsize=(14, 8)):
    """Create boxplot of tiredness index grouped by subject"""
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort subjects
    subjects = sorted(df['subject_id'].unique())
    
    # Prepare data for boxplot
    data_by_subject = [df[df['subject_id'] == subj]['tiredness_index'].values 
                       for subj in subjects]
    
    # Create boxplot
    bp = ax.boxplot(data_by_subject, 
                    labels=[f'S{int(s)}' for s in subjects],
                    patch_artist=True,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(subjects)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tiredness Index', fontsize=12, fontweight='bold')
    ax.set_title('Tiredness Index Distribution by Subject\n(All Sessions Combined)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.05, 1.05)
    
    # Add horizontal reference lines
    ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Alert threshold')
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Fatigue threshold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig, ax


def plot_boxplot_by_session(df, output_file=None, figsize=(16, 8)):
    """Create boxplot of tiredness index grouped by subject and session"""
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by subject then session
    df_sorted = df.sort_values(['subject_id', 'session_id'])
    
    # Get unique subject-session combinations
    subject_sessions = df_sorted.groupby(['subject_id', 'session_id']).size().index.tolist()
    
    # Prepare data for boxplot
    data_by_session = [
        df[(df['subject_id'] == subj) & (df['session_id'] == sess)]['tiredness_index'].values
        for subj, sess in subject_sessions
    ]
    
    # Create labels
    labels = [f'S{int(subj)}-{int(sess)}' for subj, sess in subject_sessions]
    
    # Create boxplot
    bp = ax.boxplot(data_by_session, 
                    labels=labels,
                    patch_artist=True,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=5))
    
    # Color boxes by subject
    subjects = [subj for subj, sess in subject_sessions]
    unique_subjects = sorted(set(subjects))
    subject_colors = {s: plt.cm.Set3(i/len(unique_subjects)) 
                     for i, s in enumerate(unique_subjects)}
    
    for patch, subj in zip(bp['boxes'], subjects):
        patch.set_facecolor(subject_colors[subj])
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Subject-Session', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tiredness Index', fontsize=12, fontweight='bold')
    ax.set_title('Tiredness Index Distribution by Subject and Session', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.05, 1.05)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add horizontal reference lines
    ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Alert')
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Fatigue')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig, ax


def plot_violin_by_subject(df, output_file=None, figsize=(14, 8)):
    """Create violin plot of tiredness index grouped by subject"""
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort subjects
    df_sorted = df.sort_values('subject_id')
    
    # Create violin plot
    subjects = sorted(df['subject_id'].unique())
    positions = range(1, len(subjects) + 1)
    
    parts = ax.violinplot(
        [df[df['subject_id'] == subj]['tiredness_index'].values for subj in subjects],
        positions=positions,
        showmeans=True,
        showmedians=True
    )
    
    # Color violins
    colors = plt.cm.Set3(np.linspace(0, 1, len(subjects)))
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tiredness Index', fontsize=12, fontweight='bold')
    ax.set_title('Tiredness Index Distribution by Subject\n(Violin Plot)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels([f'S{int(s)}' for s in subjects])
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.05, 1.05)
    
    # Add horizontal reference lines
    ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Alert')
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Fatigue')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig, ax


def plot_heatmap_subject_session(df, output_file=None, figsize=(12, 8)):
    """Create heatmap showing mean tiredness index for each subject-session"""
    
    # Compute mean tiredness index per subject-session
    pivot_data = df.pivot_table(
        values='tiredness_index',
        index='subject_id',
        columns='session_id',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(pivot_data, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlGn_r',  # Red (high) to Green (low)
                vmin=0, 
                vmax=1,
                cbar_kws={'label': 'Mean Tiredness Index'},
                ax=ax)
    
    ax.set_xlabel('Session', fontsize=12, fontweight='bold')
    ax.set_ylabel('Subject', fontsize=12, fontweight='bold')
    ax.set_title('Mean Tiredness Index Heatmap\n(Subject × Session)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig, ax


def print_summary_statistics(df):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nOverall Tiredness Index:")
    print(f"  Mean: {df['tiredness_index'].mean():.3f}")
    print(f"  Std:  {df['tiredness_index'].std():.3f}")
    print(f"  Min:  {df['tiredness_index'].min():.3f}")
    print(f"  Max:  {df['tiredness_index'].max():.3f}")
    print(f"  Median: {df['tiredness_index'].median():.3f}")
    
    # Per subject
    print(f"\nPer-Subject Statistics:")
    subject_stats = df.groupby('subject_id')['tiredness_index'].agg(['count', 'mean', 'std', 'min', 'max'])
    subject_stats.columns = ['N_epochs', 'Mean', 'Std', 'Min', 'Max']
    print(subject_stats.to_string())
    
    # Per session
    print(f"\nPer-Session Statistics:")
    session_stats = df.groupby(['subject_id', 'session_id'])['tiredness_index'].agg(['count', 'mean', 'std'])
    session_stats.columns = ['N_epochs', 'Mean', 'Std']
    print(session_stats.to_string())
    
    # Percentage in each category
    alert = (df['tiredness_index'] <= 0.3).sum()
    moderate = ((df['tiredness_index'] > 0.3) & (df['tiredness_index'] <= 0.7)).sum()
    fatigued = (df['tiredness_index'] > 0.7).sum()
    total = len(df)
    
    print(f"\nTiredness Categories:")
    print(f"  Alert (≤0.3):      {alert:5d} ({alert/total*100:5.1f}%)")
    print(f"  Moderate (0.3-0.7): {moderate:5d} ({moderate/total*100:5.1f}%)")
    print(f"  Fatigued (>0.7):    {fatigued:5d} ({fatigued/total*100:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize tiredness index results from CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python visualize_tiredness.py results.csv
  python visualize_tiredness.py results.csv --output-dir plots/
  python visualize_tiredness.py results.csv --plot-type all
        """
    )
    
    parser.add_argument(
        'input_csv',
        type=str,
        help='Input CSV file with tiredness index results'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='tiredness_plots',
        help='Output directory for plots (default: tiredness_plots)'
    )
    
    parser.add_argument(
        '--plot-type', '-p',
        type=str,
        choices=['boxplot-subject', 'boxplot-session', 'violin', 'heatmap', 'all'],
        default='all',
        help='Type of plot to generate (default: all)'
    )
    
    parser.add_argument(
        '--figsize',
        type=str,
        default='14,8',
        help='Figure size as "width,height" (default: 14,8)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots (only save to files)'
    )
    
    args = parser.parse_args()
    
    # Parse figsize
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
    except:
        print(f"Warning: Invalid figsize '{args.figsize}', using default (14,8)")
        figsize = (14, 8)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load data
    df = load_and_prepare_data(args.input_csv)
    
    # Print statistics
    print_summary_statistics(df)
    
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}\n")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    
    # Generate plots
    plots_to_make = []
    if args.plot_type == 'all':
        plots_to_make = ['boxplot-subject', 'boxplot-session', 'violin', 'heatmap']
    else:
        plots_to_make = [args.plot_type]
    
    for plot_type in plots_to_make:
        if plot_type == 'boxplot-subject':
            output_file = output_dir / 'boxplot_by_subject.png'
            print(f"Creating boxplot by subject...")
            plot_boxplot_by_subject(df, output_file=output_file, figsize=figsize)
            
        elif plot_type == 'boxplot-session':
            output_file = output_dir / 'boxplot_by_session.png'
            print(f"Creating boxplot by session...")
            plot_boxplot_by_session(df, output_file=output_file, figsize=(16, 8))
            
        elif plot_type == 'violin':
            output_file = output_dir / 'violin_by_subject.png'
            print(f"Creating violin plot...")
            plot_violin_by_subject(df, output_file=output_file, figsize=figsize)
            
        elif plot_type == 'heatmap':
            output_file = output_dir / 'heatmap_subject_session.png'
            print(f"Creating heatmap...")
            plot_heatmap_subject_session(df, output_file=output_file, figsize=(12, 8))
    
    print(f"\n{'='*60}")
    print(f"COMPLETE! All plots saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    # Show plots
    if not args.no_show:
        print("Displaying plots... (close windows to exit)")
        plt.show()
    else:
        plt.close('all')


if __name__ == "__main__":
    main()