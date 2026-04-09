import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

def get_latest_timestamp(raw_dir):
    """获取raw_dir/hrrr目录中最新的timestamp"""
    hrrr_dir = os.path.join(raw_dir, 'hrrr')
    
    if not os.path.exists(hrrr_dir):
        print(f"目录 {hrrr_dir} 不存在，将从开始时间下载")
        return None
    
    # 获取所有子目录（timestamp格式的目录）
    timestamp_dirs = []
    for item in os.listdir(hrrr_dir):
        item_path = os.path.join(hrrr_dir, item)
        if os.path.isdir(item_path):
            # 尝试解析为timestamp格式
            try:
                # HRRR的timestamp格式通常是 YYYYMMDDHH
                dt = datetime.strptime(item, '%Y%m%d%H')
                timestamp_dirs.append((item, dt))
            except ValueError:
                continue
    
    if not timestamp_dirs:
        print(f"目录 {hrrr_dir} 中没有找到timestamp格式的子目录，将从开始时间下载")
        return None
    
    # 按时间排序，获取最新的
    timestamp_dirs.sort(key=lambda x: x[1])
    latest_timestamp, latest_dt = timestamp_dirs[-1]
    
    print(f"找到最新的timestamp: {latest_timestamp} ({latest_dt})")
    return latest_timestamp

def get_next_timestamp(latest_timestamp, year):
    """根据最新的timestamp获取下一个时间点"""
    if latest_timestamp is None:
        return f"{year}010100"
    
    # 解析最新的timestamp
    latest_dt = datetime.strptime(latest_timestamp, '%Y%m%d%H')
    
    # 加1小时
    next_dt = latest_dt + timedelta(hours=1)
    
    # 如果跨年了，返回None表示已经下载完成
    if next_dt.year > year:
        print(f"已经下载完 {year} 年的所有数据")
        return None
    
    return next_dt.strftime('%Y%m%d%H')

def parse_timestamp_to_datetime(timestamp):
    """将timestamp字符串解析为datetime对象"""
    return datetime.strptime(timestamp, '%Y%m%d%H')

def format_datetime_to_timestamp(dt):
    """将datetime对象格式化为timestamp字符串"""
    return dt.strftime('%Y%m%d%H')

def run_download_script(script_path, year, begin_date, end_date, horizon, raw_dir, status_file, max_workers):
    """运行下载脚本"""
    cmd = [
        sys.executable, script_path,
        '--year', str(year),
        '--begin_date', begin_date,
        '--end_date', end_date,
        '--horizon', str(horizon),
        '--raw_dir', raw_dir,
        '--status_file', status_file,
        '--max_workers', str(max_workers)
    ]
    
    print(f"\n执行命令: {' '.join(cmd)}")
    print("=" * 80)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("=" * 80)
        print(f"下载脚本执行成功，返回码: {result.returncode}")
        return True
    except subprocess.CalledProcessError as e:
        print("=" * 80)
        print(f"下载脚本执行失败，返回码: {e.returncode}")
        return False
    except Exception as e:
        print("=" * 80)
        print(f"执行下载脚本时发生错误: {e}")
        return False

def check_download_status(status_file):
    """检查下载状态文件，获取完成和未完成的时间点"""
    if not os.path.exists(status_file):
        return {}, []
    
    with open(status_file, 'r') as f:
        status = json.load(f)
    
    completed = []
    incomplete = []
    
    for time_str, time_status in status.items():
        if time_status == 'completed':
            completed.append(time_str)
        else:
            incomplete.append(time_str)
    
    return status, completed, incomplete

def main():
    parser = argparse.ArgumentParser(description='HRRR数据下载控制器 - 支持断点续传和多次重试')
    parser.add_argument('--year', type=int, default=2020, help='Year to download')
    parser.add_argument('--begin_date', type=str, default='0207', help='Begin date in MMDD format')
    parser.add_argument('--end_date', type=str, default='1231', help='End date in MMDD format')
    parser.add_argument('--horizon', type=int, default=0, help='Forecast horizon')
    parser.add_argument('--raw_dir', type=str, default='/hpc2hdd/home/hni017/Workplace/ExtremeWeather/weather_data/HRRR/raw', help='Raw data directory')
    parser.add_argument('--download_script', type=str, default='download_raw.py', help='Download script path')
    parser.add_argument('--status_file', type=str, default='download_status.json', help='Status file path')
    parser.add_argument('--max_workers', type=int, default=96, help='Maximum number of workers')
    parser.add_argument('--max_retries', type=int, default=10, help='Maximum number of retries when download fails')
    parser.add_argument('--retry_delay', type=int, default=60, help='Delay between retries in seconds')
    parser.add_argument('--check_interval', type=int, default=300, help='Interval between status checks in seconds')
    parser.add_argument('--auto_restart', action='store_true', help='Automatically restart download when it fails')
    parser.add_argument('--resume_from_latest', action='store_true', help='Resume from the latest timestamp in raw_dir/hrrr')
    
    args = parser.parse_args()
    
    # 确保下载脚本路径是绝对路径
    if not os.path.isabs(args.download_script):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.download_script = os.path.join(script_dir, args.download_script)
    
    print("=" * 80)
    print("HRRR数据下载控制器")
    print("=" * 80)
    print(f"年份: {args.year}")
    print(f"开始日期: {args.begin_date}")
    print(f"结束日期: {args.end_date}")
    print(f"预报时效: {args.horizon}")
    print(f"数据目录: {args.raw_dir}")
    print(f"下载脚本: {args.download_script}")
    print(f"状态文件: {args.status_file}")
    print(f"最大工作线程: {args.max_workers}")
    print(f"最大重试次数: {args.max_retries}")
    print(f"重试延迟: {args.retry_delay}秒")
    print(f"状态检查间隔: {args.check_interval}秒")
    print(f"自动重启: {'是' if args.auto_restart else '否'}")
    print(f"从最新timestamp恢复: {'是' if args.resume_from_latest else '否'}")
    print("=" * 80)
    
    # 如果需要从最新timestamp恢复
    if args.resume_from_latest:
        latest_timestamp = get_latest_timestamp(args.raw_dir)
        if latest_timestamp:
            latest_dt = parse_timestamp_to_datetime(latest_timestamp)
            next_dt = latest_dt + timedelta(hours=1)
            
            # 检查是否已经下载完成
            end_dt = datetime.strptime(f"{args.year}{args.end_date}23", '%Y%m%d%H')
            if next_dt > end_dt:
                print(f"所有数据已下载完成！")
                return
            
            # 更新开始日期
            new_begin_date = next_dt.strftime('%m%d')
            print(f"从最新timestamp恢复，新的开始日期: {new_begin_date}")
            args.begin_date = new_begin_date
    
    # 执行下载循环
    retry_count = 0
    while retry_count <= args.max_retries:
        print(f"\n开始第 {retry_count + 1} 次下载尝试...")
        print("-" * 80)
        
        # 运行下载脚本
        success = run_download_script(
            args.download_script,
            args.year,
            args.begin_date,
            args.end_date,
            args.horizon,
            args.raw_dir,
            args.status_file,
            args.max_workers
        )
        
        if success:
            print("\n下载成功完成！")
            
            # 检查下载状态
            status, completed, incomplete = check_download_status(args.status_file)
            print(f"已完成时间点: {len(completed)}")
            print(f"未完成时间点: {len(incomplete)}")
            
            if not incomplete:
                print("所有时间点都已下载完成！")
                break
            else:
                print(f"还有 {len(incomplete)} 个时间点未完成")
                if not args.auto_restart:
                    print("请手动重新运行以继续下载未完成的时间点")
                    break
                else:
                    print("自动重启以继续下载...")
        else:
            print(f"\n下载失败 (尝试 {retry_count + 1}/{args.max_retries})")
            
            if retry_count < args.max_retries:
                print(f"等待 {args.retry_delay} 秒后重试...")
                time.sleep(args.retry_delay)
            else:
                print("已达到最大重试次数，停止尝试")
        
        retry_count += 1
    
    print("\n" + "=" * 80)
    print("下载控制器结束")
    print("=" * 80)

if __name__ == '__main__':
    main()
