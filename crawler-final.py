import time
import json
import os
import re
import random
from DrissionPage import ChromiumPage

class JDCommentSpider:
    def __init__(self):
        # 商品ID池
        self.product_ids = [
            # 3C数码与家电
            '10108323353757',   # 键盘
            '100209267857',     # 手机1
            '10199734921580',   # 手机2
            '100133134399',     # 空调1
            '100206912345',     # 空调2
            '10111029693928',   # 主机1
            '100243877083',     # 主机2

            # 服饰鞋包与配饰
            '10154346890183',   # 背包1
            '100266194006',     # 背包2
            '10211232299817',   # 短袖1
            '100234974129',     # 短袖2
            '100269110512',     # 手链
            '10214963804591',   # 项链
            '100122754815',     # 帽子

            # 美妆个护与健康
            '100216909178',     # 防晒1
            '100166610871',     # 防晒2
            '100117948314',     # 面霜1
            '100238431546',     # 面霜2

            # 家居生活与家具
            '100121506856',     # 床垫1
            '100113877112',     # 床垫2
            '100093395587',     # 四件套1
            '100209368108',     # 四件套2

            # 食品饮料与生鲜
            '100012043978',     # 酒1
            '100003033647',     # 酒2
            '100045078940',     # 方便面1
            '100028121650',     # 方便面2
            '10155933569002',   # 水果1
            '10150116508367',   # 水果2

            # 母婴用品与玩具
            '10171932842182',   # 玩具1
            '10209539375803',   # 玩具2

            # 运动户外与器材
            '10140340235145',     # 运动器材1
            '10184213590364',     # 运动器材2
            '100167785051',     # 运动器材3

            # 图书文具与办公
            '10118799465329',  # 书籍1
            '13443715',  # 书籍2
            '100121416221',  # 办公用品1
            '6083519',  # 办公用品2
        ]
        
        # 创建保存目录
        self.save_dir = 'goods'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 浏览器实例
        self.dp = None
    
    def clean_filename(self, text):
        """清理文件名中的非法字符"""
        illegal_chars = r'[<>:"/\\|?*]'
        cleaned = re.sub(illegal_chars, '_', text)
        if len(cleaned) > 50:
            cleaned = cleaned[:50]
        return cleaned.strip()
    
    def get_product_title_from_title(self):
        """从页面<title>标签中获取商品名称"""
        try:
            page_title = self.dp.title
            print(f"  📌 原始页面标题: {page_title}")
            
            if not page_title:
                return "未知商品"
            
            # 去除【】内的内容
            title_clean = re.sub(r'【[^】]*】', '', page_title)
            
            # 去除常见后缀
            suffixes = [
                '京东价', '多少钱', '怎么样', '正品保证', '报价', '价格', 
                '评价', '怎么样？', '好不好', '参数', '配置', '图片',
                '【', '】', '京东自营', '官方旗舰店'
            ]
            
            for suffix in suffixes:
                if suffix in title_clean:
                    title_clean = title_clean.split(suffix)[0]
            
            title_clean = title_clean.strip()
            
            if '京东' in title_clean:
                title_clean = title_clean.replace('京东', '').strip()
            
            if not title_clean or len(title_clean) < 2:
                title_clean = page_title[:30].strip()
            
            print(f"  📦 提取的商品名称: {title_clean}")
            return title_clean
            
        except Exception as e:
            print(f"  ⚠️ 获取商品标题失败: {str(e)}")
            return "未知商品"
    
    def get_total_comment_count(self):
        """获取商品的总评论数"""
        try:
            # 尝试多种选择器获取总评论数
            selectors = [
                'css:.comment-count',
                'css:.J-comment-count',
                'css:._commentCount',
                'xpath://span[contains(@class, "comment") and contains(text(), "条评论")]'
            ]
            
            for selector in selectors:
                try:
                    if selector.startswith('css:'):
                        count_elem = self.dp.ele(selector, timeout=2)
                    else:
                        count_elem = self.dp.ele(selector, timeout=2)
                    
                    if count_elem:
                        count_text = count_elem.text
                        numbers = re.findall(r'\d+', count_text)
                        if numbers:
                            total = int(numbers[0])
                            print(f"  📊 商品总评论数: {total}")
                            return total
                except:
                    continue
            
            print(f"  ⚠️ 未能获取总评论数，将使用目标数量70条")
            return None
        except Exception as e:
            print(f"  ⚠️ 获取总评论数失败: {str(e)}")
            return None
    
    def init_browser(self):
        """初始化浏览器"""
        if self.dp is None:
            self.dp = ChromiumPage()
            print("✅ 浏览器初始化完成")
    
    def close_browser(self):
        """关闭浏览器"""
        if self.dp:
            self.dp.quit()
            self.dp = None
            print("✅ 浏览器已关闭")
    
    def get_comments(self, product_id, target_count=70):
        """爬取单个商品的评论（持续滚动浮窗直到达到目标数量）"""
        url = f'https://item.jd.com/{product_id}.html'
        print(f"\n{'='*60}")
        print(f"🔄 开始爬取商品: {product_id}")
        print(f"🎯 目标评论数: {target_count}")
        print(f"📎 商品链接: {url}")
        
        try:
            # 访问商品页面
            self.dp.get(url)
            time.sleep(random.uniform(3, 8))
            
            # 获取商品名称
            product_title = self.get_product_title_from_title()
            
            # 尝试获取实际总评论数
            total_comments = self.get_total_comment_count()
            if total_comments:
                actual_target = min(target_count, total_comments)
                print(f"🎯 实际目标评论数: {actual_target} (min(目标{target_count}, 总评论{total_comments}))")
            else:
                actual_target = target_count
                print(f"🎯 实际目标评论数: {actual_target}")
            
            print(f"{'='*60}")
            
            # 开始监听数据包
            self.dp.listen.start('client.action')
            
            # 点击"全部评价"按钮
            btns = self.dp.eles('css:.all-btn')
            clicked = False
            for btn in btns:
                if btn.text == '全部评价':
                    btn.click()
                    print("✅ 成功点击'全部评价'按钮")
                    clicked = True
                    break
            
            if not clicked:
                print("❌ 未找到'全部评价'按钮")
                return [], product_title
            
            # 等待评论浮窗加载
            time.sleep(3)
            
            # 查找评论浮窗容器（关键：需要滚动这个容器）
            comment_container = None
            
            # 尝试多种选择器找到评论列表容器
            selectors = [
                'css:.comment-list',
                'css:.J-comment-list',
                'css:._comment-list',
                'css:.mod-rate-list',
                'xpath://div[contains(@class, "comment") and contains(@class, "list")]',
                'css:[class*="commentList"]',
                'css:[class*="rateList"]'
            ]
            
            for selector in selectors:
                try:
                    if selector.startswith('css:'):
                        container = self.dp.ele(selector, timeout=2)
                    else:
                        container = self.dp.ele(selector, timeout=2)
                    
                    if container:
                        comment_container = container
                        print(f"✅ 找到评论容器: {selector}")
                        break
                except:
                    continue
            
            # 如果找不到特定容器，使用整个页面作为备选
            if not comment_container:
                print(f"⚠️ 未找到评论容器，将滚动整个页面")
                comment_container = self.dp
            
            # 存储评论（使用字典去重）
            all_comments = []
            seen_comment_ids = set()  # 用于去重
            
            page = 1
            no_new_data_count = 0  # 连续无新数据的次数
            last_comment_count = 0
            max_no_new_data = 1  # 最大连续无新数据次数
            
            # 持续加载直到达到目标数量或没有新评论
            # 持续加载直到达到目标数量或没有新评论
            while len(all_comments) < actual_target and no_new_data_count < max_no_new_data:
                print(f'  📄 正在加载第{page}页评论... (当前: {len(all_comments)}/{actual_target})')
                
                try:
                    resp = self.dp.listen.wait(timeout=15)
                    json_data = resp.response.body

                    # 安全提取评论数据
                    comment_list = None
                    try:
                        # 尝试原路径：result.floors[2].data
                        floors = json_data.get('result', {}).get('floors')
                        if isinstance(floors, list) and len(floors) > 2:
                            comment_list = floors[2].get('data')
                        elif isinstance(floors, dict):
                            # 如果floors是字典，尝试获取键为'2'或'commentList'等
                            comment_list = floors.get('2', {}).get('data') or floors.get('commentList')
                        else:
                            # 其他可能的路径
                            comment_list = json_data.get('result', {}).get('commentList')
                            if not comment_list:
                                comment_list = json_data.get('data', {}).get('comments')
                    except Exception as e:
                        print(f"    ⚠️ 解析数据结构失败: {e}")

                    # 如果仍然没有找到评论列表，尝试直接搜索commentInfo
                    if not comment_list:
                        # 递归搜索所有键值对，寻找包含'commentInfo'的列表
                        def find_comment_list(obj, depth=0):
                            if depth > 5:
                                return None
                            if isinstance(obj, dict):
                                for k, v in obj.items():
                                    if k == 'commentInfo' and isinstance(v, dict):
                                        # 如果找到单个commentInfo，包装成列表
                                        return [obj]
                                    res = find_comment_list(v, depth+1)
                                    if res:
                                        return res
                            elif isinstance(obj, list):
                                for item in obj:
                                    if isinstance(item, dict) and 'commentInfo' in item:
                                        return obj  # 直接返回整个列表
                                    res = find_comment_list(item, depth+1)
                                    if res:
                                        return res
                            return None
                        comment_list = find_comment_list(json_data)

                    if not comment_list:
                        print(f"    ⚠️ 无法从数据包中提取评论列表，跳过该页")
                        continue  # 跳过本次循环，继续等待下一个包

                    # 确保comment_list是列表
                    if not isinstance(comment_list, list):
                        comment_list = [comment_list] if comment_list else []

                    page_new_comments = 0
                    for comment_item in comment_list:
                        # 兼容两种结构：直接包含commentInfo，或者commentInfo在外层
                        comment_info = comment_item.get('commentInfo') if 'commentInfo' in comment_item else comment_item
                        if comment_info and isinstance(comment_info, dict):
                            comment_id = f"{comment_info.get('commentData', '')}_{comment_info.get('commentDate', '')}_{comment_info.get('userNickName', '')}"
                            if comment_id not in seen_comment_ids:
                                seen_comment_ids.add(comment_id)
                                score_raw = comment_info.get('commentScore', 0)
                                try:
                                    score_value = int(score_raw)
                                except (ValueError, TypeError):
                                    score_value = 0
                                comment = {
                                    '商品ID': product_id,
                                    '商品名称': product_title,
                                    '昵称': comment_info.get('userNickName', ''),
                                    '购买产品': comment_info.get('productSpecifications', '').replace('已购', ''),
                                    '购买次数': comment_info.get('buyCount', ''),
                                    '评论内容': comment_info.get('commentData', ''),
                                    '评论时间(日期)': comment_info.get('commentDate', ''),
                                    '评分': score_value,
                                    '用户等级': comment_info.get('userLevelName', ''),
                                    '回复数': comment_info.get('replyCount', 0),
                                    '点赞数': comment_info.get('supportCount', 0)
                                }
                                all_comments.append(comment)
                                page_new_comments += 1

                    print(f'    ✅ 第{page}页加载到 {page_new_comments} 条新评论 (累计: {len(all_comments)})')
                    
                    # 检查是否有新数据
                    if len(all_comments) == last_comment_count:
                        no_new_data_count += 1
                        print(f'    ⚠️ 未检测到新评论 (连续{no_new_data_count}/{max_no_new_data}次)')
                    else:
                        no_new_data_count = 0
                        last_comment_count = len(all_comments)
                    
                    # 如果已达到目标数量，提前结束
                    if len(all_comments) >= actual_target:
                        print(f'    🎉 已达到目标数量 {actual_target} 条评论，停止加载')
                        break
                    
                    # 滚动浮窗容器
                    for scroll_count in range(3):
                        comment_container.scroll.to_bottom()
                        time.sleep(0.5)
                    
                    time.sleep(2)
                    page += 1
                    
                except Exception as e:
                    print(f"    ❌ 第{page}页加载失败: {str(e)}")
                    print(f"    📌 出现失败，停止继续爬取，保存当前已有的 {len(all_comments)} 条评论")
                    break  # 直接退出循环，不再重试
            
            # 输出最终统计
            print(f"\n📊 商品 {product_id} 采集完成:")
            print(f"   实际采集评论数: {len(all_comments)}")
            print(f"   目标评论数: {actual_target}")
            
            if len(all_comments) < actual_target:
                print(f"   ⚠️ 未达到目标数量，可能商品评论总数不足 {actual_target} 条")
            
            return all_comments, product_title
            
        except Exception as e:
            print(f"❌ 商品 {product_id} 爬取失败: {str(e)}")
            return [], "未知商品"
    
    def save_comments(self, product_id, product_title, comments):
        """保存评论到JSON文件"""
        if not comments:
            print(f"⚠️ 商品 {product_id} 无评论数据，跳过保存")
            return False
        
        # 清理商品名称中的非法字符
        clean_title = self.clean_filename(product_title)
        
        # 生成文件名（商品ID_商品名称_时间戳.json）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{product_id}_{clean_title}_{timestamp}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        # 保存数据
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comments, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 商品数据保存成功:")
        print(f"   商品ID: {product_id}")
        print(f"   商品名称: {product_title}")
        print(f"   文件路径: {filepath}")
        print(f"   评论数量: {len(comments)}")
        
        # 显示统计信息
        if comments:
            scores = []
            for c in comments:
                score = c.get('评分', 0)
                if score:
                    try:
                        scores.append(int(score))
                    except (ValueError, TypeError):
                        continue
            
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"   平均评分: {avg_score:.2f}")
                
                # 评分分布
                score_dist = {}
                for score in scores:
                    score_dist[score] = score_dist.get(score, 0) + 1
                print(f"   评分分布: {score_dist}")
            else:
                print(f"   ⚠️ 无有效评分数据")
        
        return True
    
    def run(self, target_count=40, wait_time=70, batch_rest=3, batch_rest_minutes=3):
        """运行爬虫
        
        Args:
            target_count: 每个商品目标评论数
            wait_time: 商品间等待时间（秒）
            batch_rest: 每爬取多少个商品后休息
            batch_rest_minutes: 批量休息时间（分钟）
        """
        print("🚀 京东评论爬虫启动")
        print(f"📦 待爬取商品数量: {len(self.product_ids)}")
        print(f"🎯 每个商品目标评论数: {target_count}")
        print(f"⏱️  商品间隔时间: {wait_time}秒")
        print(f"🛌 批量休息策略: 每{batch_rest}个商品休息{batch_rest_minutes}分钟")
        print("-" * 60)
        
        # 初始化浏览器
        self.init_browser()
        
        # 统计信息
        total_comments = 0
        success_count = 0
        
        # 存储所有商品信息（用于最后展示）
        products_info = []
        
        # 遍历商品ID池
        for idx, product_id in enumerate(self.product_ids, 1):
            print(f"\n{'#'*60}")
            print(f"进度: [{idx}/{len(self.product_ids)}]")
            
            # 爬取商品评论
            comments, product_title = self.get_comments(product_id, target_count)
            
            # 保存数据
            if self.save_comments(product_id, product_title, comments):
                success_count += 1
                total_comments += len(comments)
                products_info.append({
                    'id': product_id,
                    'title': product_title,
                    'comments_count': len(comments)
                })
            else:
                print(f"❌ 商品 {product_id} 采集失败")
            
            # 等待时间（最后一个商品不需要等待）
            if idx < len(self.product_ids):
                print(f"\n⏰ 等待 {wait_time} 秒后继续下一个商品...")
                time.sleep(wait_time)
            
            # 每爬取 batch_rest 个商品后额外休息
            if idx % batch_rest == 0 and idx < len(self.product_ids):
                rest_seconds = batch_rest_minutes * 60
                print(f"\n🛑 已完成 {idx} 个商品，为防止监测，休息{batch_rest_minutes}分钟（{rest_seconds}秒）...")
                time.sleep(rest_seconds)
                print("✅ 休息结束，继续爬取下一个商品")
        
        # 输出最终统计
        print(f"\n{'='*60}")
        print("📊 爬取完成统计:")
        print(f"   成功商品数: {success_count}/{len(self.product_ids)}")
        print(f"   总评论数: {total_comments}")
        print(f"   平均每商品评论数: {total_comments//success_count if success_count else 0}")
        print(f"   保存目录: {self.save_dir}")
        print(f"\n📋 商品详情:")
        for info in products_info:
            # 截断过长的标题用于显示
            display_title = info['title'][:40] + '...' if len(info['title']) > 40 else info['title']
            print(f"   - {info['id']}: {display_title} ({info['comments_count']}条评论)")
        print(f"{'='*60}")
        
        # 关闭浏览器
        self.close_browser()
    
    def run_single(self, product_id, target_count=70, max_pages=None):
        """测试单个商品
        
        Args:
            product_id: 商品ID
            target_count: 目标评论数
            max_pages: 最大页数（向后兼容，实际不使用）
        """
        print("🚀 单商品测试模式")
        self.init_browser()
        comments, product_title = self.get_comments(product_id, target_count)
        self.save_comments(product_id, product_title, comments)
        self.close_browser()
        return comments

# 使用示例
if __name__ == '__main__':
    # 创建爬虫实例
    spider = JDCommentSpider()
    
    # 批量爬取所有商品（每商品70条评论，每4个商品休息3分钟）
    spider.run(target_count=70, wait_time=random.randint(60, 120), batch_rest=2, batch_rest_minutes=random.randint(3, 5))
    
    # 方式2：测试单个商品
    # spider.run_single('100012043978', target_count=50)
    
    # 方式3：自定义商品ID池
    # spider.product_ids = ['100012043978', '10108323353757']
    # spider.run(target_count=50, wait_time=30, batch_rest=3, batch_rest_minutes=2)