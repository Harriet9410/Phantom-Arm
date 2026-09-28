/*

* 2025无人系统具身智能算法挑战赛 专用代码
* 版权所有 (c) 2025 无人系统具身智能算法挑战赛组委会
* 
* 本源码仅限本赛事参赛团队在比赛过程中使用，
* 禁止任何形式的商业用途、非授权传播或用于其他非比赛场景。
* 
* 依照 GNU 通用公共许可证（GPL）条款授权：
* 参赛者可基于赛事目的对源码进行修改和扩展，
* 但修改后的代码仍受限于本声明的约束条款。
* 
* 本源码按"现状"提供，组委会不承担任何明示或暗示的担保责任，
* 包括但不限于适销性或特定用途适用性的保证。

*/
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <cmath>
#include <mutex>
#include <thread>

class PickAndPlaceNode
{
public:
    PickAndPlaceNode(ros::NodeHandle &nh);

private:
    void sideCallback(const std_msgs::String::ConstPtr &msg);
    void endEffectorCallback(const geometry_msgs::PoseStamped::ConstPtr &msg);
    void arrayCallback(const std_msgs::Float64MultiArray::ConstPtr &msg);
    void gripperCapturedCallback(const std_msgs::Bool::ConstPtr &msg);

    void publishPose(const geometry_msgs::PoseStamped &ps, const std::string &tag);
    void controlGripperIncremental(bool closed, const std::string &tag);
    void openGripperDirectly(const std::string &tag);
    void waitForPosition(const geometry_msgs::PoseStamped &target);
    void waitForGripper(const geometry_msgs::PoseStamped &hold);
    void gripperEffortCallback(const std_msgs::Float32MultiArray::ConstPtr &msg);
    void doPickPlace(geometry_msgs::PoseStamped target);
    double calcDistance(const geometry_msgs::PoseStamped &a, const geometry_msgs::PoseStamped &b);

    ros::NodeHandle nh_;
    ros::Publisher ee_pose_pub_, gripper_pub_, result_pub_;
    ros::Subscriber side_sub_, end_pose_sub_, object_sub_, gripper_sub_, effort_sub_;

    geometry_msgs::PoseStamped place_pose_, init_pose_, current_pose_;
    bool gripper_closed_, busy_, got_end_pose_, pick_success_, grasped_;
    std::string frame_id_, conveyor_side_;
    double approach_height_, left_place_x_, right_place_x_;
    double pos_tolerance_, ori_tolerance_, max_step_distance_;
    double grasp_z_offset_;
    float gripper_value_, gripper_step_, grasp_value_;
    float live_effort_, grasp_effort_, lift_effort_;
    std::mutex mutex_, pose_mutex_;
};

PickAndPlaceNode::PickAndPlaceNode(ros::NodeHandle &nh)
    : nh_(nh), busy_(false), got_end_pose_(false),
      gripper_closed_(false), pick_success_(false), grasped_(false),
      grasp_value_(0.0f), live_effort_(0.0f), grasp_effort_(0.0f),
      lift_effort_(0.0f),
      gripper_value_(0.0f), gripper_step_(0.05f),
      conveyor_side_("left")
{
    ros::NodeHandle pnh("~");
    pnh.param<std::string>("frame_id", frame_id_, "base_link");
    pnh.param("approach_height", approach_height_, 0.1);
    pnh.param("grasp_z_offset", grasp_z_offset_, -0.04);
    pnh.param("pos_tolerance", pos_tolerance_, 0.06);
    pnh.param("ori_tolerance", ori_tolerance_, 0.2);
    pnh.param("max_step_distance", max_step_distance_, 0.5);
    pnh.param("place_pose_x", left_place_x_, 0.60);
    pnh.param("place_pose_y", place_pose_.pose.position.y, 0.5);
    pnh.param("place_pose_z", place_pose_.pose.position.z, 2.67);
    pnh.param("place_pose_qx", place_pose_.pose.orientation.x, 0.00);
    pnh.param("place_pose_qy", place_pose_.pose.orientation.y, 1.2);
    pnh.param("place_pose_qz", place_pose_.pose.orientation.z, 1.2);
    pnh.param("place_pose_qw", place_pose_.pose.orientation.w, 0.0);
    // [修复B4] place_pose 的四元数 (0,1.2,1.2,0) 模长 1.697，不是单位四元数；
    // angleShortestPath 按单位四元数定义，不归一化会让姿态误差算错。
    {
        tf2::Quaternion qn(place_pose_.pose.orientation.x, place_pose_.pose.orientation.y,
                           place_pose_.pose.orientation.z, place_pose_.pose.orientation.w);
        qn.normalize();
        place_pose_.pose.orientation.x = qn.x();
        place_pose_.pose.orientation.y = qn.y();
        place_pose_.pose.orientation.z = qn.z();
        place_pose_.pose.orientation.w = qn.w();
        ROS_INFO("[修复B4] 放置姿态四元数归一化 → (%.4f, %.4f, %.4f, %.4f)",
                 qn.x(), qn.y(), qn.z(), qn.w());
    }
    place_pose_.header.frame_id = frame_id_;
    right_place_x_ = -left_place_x_;

    init_pose_.header.frame_id = frame_id_;
    init_pose_.pose.position.x = 0.0662943895;
    init_pose_.pose.position.y = 0.0862721270;
    init_pose_.pose.position.z = 2.7503792615;
    init_pose_.pose.orientation.x = 0.0146265891;
    init_pose_.pose.orientation.y = -0.0000924757;
    init_pose_.pose.orientation.z = 0.9998736759;
    init_pose_.pose.orientation.w = -0.0062198575;

    ee_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/Jaka/set_end_effector_pose", 10);
    gripper_pub_ = nh_.advertise<std_msgs::Float32>("/Jaka/set_gripper_value", 10);
    result_pub_ = nh_.advertise<std_msgs::Bool>(
        "/pick_place_result", 1,
        /*latched=*/true);

    side_sub_ = nh_.subscribe("/conveyor_side", 10, &PickAndPlaceNode::sideCallback, this);
    object_sub_ = nh_.subscribe("/arm_end_pose_quaternion", 10, &PickAndPlaceNode::arrayCallback, this);
    end_pose_sub_ = nh_.subscribe("/Jaka/get_end_effector_pose", 10, &PickAndPlaceNode::endEffectorCallback, this);
    gripper_sub_ = nh_.subscribe("/Jaka/gripper_is_captured", 10, &PickAndPlaceNode::gripperCapturedCallback, this);
    effort_sub_ = nh_.subscribe("/Jaka/get_gripper_efforts", 10, &PickAndPlaceNode::gripperEffortCallback, this);

    ros::Duration(1.0).sleep();
    openGripperDirectly("夹爪初始化");
    ROS_INFO("[PickAndPlaceNode] 夹爪初始化完成，已张开");
    ROS_INFO("[PickAndPlaceNode] 初始化完成");
}

void PickAndPlaceNode::sideCallback(const std_msgs::String::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(mutex_);
    conveyor_side_ = msg->data;
    ROS_INFO("收到 /conveyor_side: %s", conveyor_side_.c_str());
}

void PickAndPlaceNode::endEffectorCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(pose_mutex_);
    current_pose_ = *msg;
    got_end_pose_ = true;
}

void PickAndPlaceNode::gripperCapturedCallback(const std_msgs::Bool::ConstPtr &msg)
{
    gripper_closed_ = msg->data;
}

// [修复B6] 记录闭合时的最大手指受力。"接触信号"在 jaka_env 里由指令越过 0.03
// 触发（force-attach），不代表真的夹住了东西；受力才是真凭据。
void PickAndPlaceNode::gripperEffortCallback(const std_msgs::Float32MultiArray::ConstPtr &msg)
{
    float m = 0.0f;
    for (size_t i = 0; i < msg->data.size(); ++i)
    {
        float v = (float)std::fabs(msg->data[i]);
        if (std::isfinite(v) && v > m) m = v;
    }
    live_effort_ = m;
}

void PickAndPlaceNode::arrayCallback(const std_msgs::Float64MultiArray::ConstPtr &msg)
{
    if (msg->data.size() < 7)
    {
        ROS_WARN_THROTTLE(5.0, "数据长度 %lu < 7，跳过", msg->data.size());
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (busy_)
            return;
        busy_ = true;
        pick_success_ = false;
    }

    geometry_msgs::PoseStamped target;
    target.header.stamp = ros::Time::now();
    target.header.frame_id = frame_id_;
    target.pose.position.x = msg->data[0];
    target.pose.position.y = msg->data[1];
    target.pose.position.z = msg->data[2];
    target.pose.orientation.x = msg->data[3];
    target.pose.orientation.y = msg->data[4];
    target.pose.orientation.z = msg->data[5];
    target.pose.orientation.w = msg->data[6];

    std::thread(&PickAndPlaceNode::doPickPlace, this, target).detach();
}

void PickAndPlaceNode::publishPose(const geometry_msgs::PoseStamped &ps, const std::string &tag)
{
    ee_pose_pub_.publish(ps);
    ROS_INFO("%s → [x=%.3f, y=%.3f, z=%.3f]", tag.c_str(),
             ps.pose.position.x, ps.pose.position.y, ps.pose.position.z);
}

void PickAndPlaceNode::controlGripperIncremental(bool closed, const std::string &tag)
{
    // [修复B5] jaka_env 把这条指令当"手指关节位置"用：
    //     self._gripper_open = 0.0 / self._gripper_closed = 0.001
    // 可用行程约 0~0.04（A 方案实测范围，步长 0.00025）。
    // 原版 gripper_step_=0.05、上界 1.0 —— 第一次下发 0.05 就冲出量程。
    const float kOpen = 0.0f;
    const float kGripMax = 0.04f;
    const float kGripStep = 0.001f;
    gripper_value_ = closed
                         ? std::min(kGripMax, gripper_value_ + kGripStep)
                         : std::max(kOpen, gripper_value_ - kGripStep);
    std_msgs::Float32 cmd;
    cmd.data = gripper_value_;
    gripper_pub_.publish(cmd);
    ROS_INFO("%s → 夹爪值: %.4f", tag.c_str(), gripper_value_);
}

void PickAndPlaceNode::openGripperDirectly(const std::string &tag)
{
    // [修复B5] jaka_env 的张开值是 0.0，不是 0.001。
    // 原版初始化时发 0.001 并打印"已张开"，实际下发的是闭合侧的数值。
    gripper_value_ = 0.0f;
    std_msgs::Float32 cmd;
    cmd.data = gripper_value_;
    gripper_pub_.publish(cmd);
    ROS_INFO("%s → 夹爪直接设置为: %.4f（张开）", tag.c_str(), gripper_value_);
}

double PickAndPlaceNode::calcDistance(const geometry_msgs::PoseStamped &a,
                                      const geometry_msgs::PoseStamped &b)
{
    double dx = a.pose.position.x - b.pose.position.x;
    double dy = a.pose.position.y - b.pose.position.y;
    double dz = a.pose.position.z - b.pose.position.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void PickAndPlaceNode::waitForPosition(const geometry_msgs::PoseStamped &target)
{
    ros::Rate rate(20);
    int tries = 0;
    while (ros::ok())
    {
        if (++tries > 500) { ROS_WARN("POS-WAIT-TIMEOUT 25s, force continue"); break; }
        geometry_msgs::PoseStamped cmd;
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            if (!got_end_pose_)
            {
                ros::spinOnce();
                rate.sleep();
                continue;
            }

            double dist = calcDistance(current_pose_, target);
            tf2::Quaternion qc(current_pose_.pose.orientation.x,
                               current_pose_.pose.orientation.y,
                               current_pose_.pose.orientation.z,
                               current_pose_.pose.orientation.w);
            tf2::Quaternion qt(target.pose.orientation.x,
                               target.pose.orientation.y,
                               target.pose.orientation.z,
                               target.pose.orientation.w);
            double angle = qc.angleShortestPath(qt);
            if (dist <= pos_tolerance_ && angle <= ori_tolerance_)
            {
                ROS_INFO("目标到达 (误差: %.3fm, %.2f°)", dist, angle * 180.0 / M_PI);
                break;
            }

            cmd = target;
            if (dist > max_step_distance_)
            {
                double ratio = max_step_distance_ / dist;
                cmd.pose.position.x = current_pose_.pose.position.x + (target.pose.position.x - current_pose_.pose.position.x) * ratio;
                cmd.pose.position.y = current_pose_.pose.position.y + (target.pose.position.y - current_pose_.pose.position.y) * ratio;
                cmd.pose.position.z = current_pose_.pose.position.z + (target.pose.position.z - current_pose_.pose.position.z) * ratio;
            }
        }
        cmd.header.stamp = ros::Time::now();
        cmd.header.frame_id = frame_id_;
        ee_pose_pub_.publish(cmd);
        ros::spinOnce();
        rate.sleep();
    }
}

void PickAndPlaceNode::waitForGripper(const geometry_msgs::PoseStamped &hold)
{
    // [修复B7] 本仿真里 /Jaka/gripper_is_captured 表示"夹爪本体撞到东西"，
    // 不是"指缝夹住物体"（见 jaka_env.py 的 contact 判定）。
    // 原版一收到它就退出 —— 实测退出时指令值只有 0.0020，手指几乎没动，
    // 所以只是"碰到了"而拿不住。现在：接触信号只记录，闭合一路做到底。
    ros::Rate rate(10);
    int tries = 0;
    bool saw_contact = false;
    const float kGripMax = 0.04f;
    while (ros::ok())
    {
        if (++tries > 80) { ROS_WARN("GRIP-WAIT-TIMEOUT 8s, force continue"); break; }
        // [修复B6] 闭合期间每一拍都重新下发目标位姿，压住腕部漂移。
        geometry_msgs::PoseStamped hold_cmd = hold;
        hold_cmd.header.stamp = ros::Time::now();
        hold_cmd.header.frame_id = frame_id_;
        ee_pose_pub_.publish(hold_cmd);
        if (gripper_closed_ && !saw_contact)
        {
            saw_contact = true;
            ROS_INFO("检测到接触信号（指令值 %.4f）—— 仅记录，继续合到行程底", gripper_value_);
        }
        if (gripper_value_ >= kGripMax - 1e-6f) break;
        controlGripperIncremental(true, "夹爪闭合");
        ros::spinOnce();
        rate.sleep();
    }
    // 合到底后保持 0.8 秒，让摩擦夹持力建立起来，再交给调用方抬臂。
    for (int i = 0; i < 8 && ros::ok(); ++i)
    {
        geometry_msgs::PoseStamped hold_cmd = hold;
        hold_cmd.header.stamp = ros::Time::now();
        hold_cmd.header.frame_id = frame_id_;
        ee_pose_pub_.publish(hold_cmd);
        std_msgs::Float32 hold_grip;
        hold_grip.data = gripper_value_;
        gripper_pub_.publish(hold_grip);
        ros::spinOnce();
        rate.sleep();
    }
    ROS_INFO("夹爪闭合结束 → 夹爪值: %.4f，接触到 %s", gripper_value_,
             saw_contact ? "有" : "无");
}

void PickAndPlaceNode::doPickPlace(geometry_msgs::PoseStamped target)
{
    // [修复B1] gripper_closed_ 是订阅回调写入的粘滞状态，原版从不复位。
    // 一旦上一轮变成 true，本轮 waitForGripper 第一次进循环就退出（只到 0.05）。
    gripper_closed_ = false;
    grasped_ = false;
    grasp_value_ = 0.0f;
    grasp_effort_ = 0.0f;
    lift_effort_ = 0.0f;
    // [修复B5] 张开值是 0.0
    gripper_value_ = 0.0f;
    {
        std_msgs::Float32 reset_cmd;
        reset_cmd.data = gripper_value_;
        gripper_pub_.publish(reset_cmd);
    }
    ROS_INFO("[修复B1/B5] 本轮开始：夹爪复位为张开 %.4f", gripper_value_);
    // [修复B4] 原版抬升点与抓取点在 x 上差 0.02m，两点间的"下移"其实是斜线；
    // 横移会让求解器重算姿态，实测在抓取点姿态误差达 53~61°。
    // 改成：抬升点就放在抓取点正上方，下移走纯 Z 方向。
    target.pose.position.x += 0.01;
    target.pose.position.y += 0.0075;
    target.pose.position.z += grasp_z_offset_;

    geometry_msgs::PoseStamped approach = target;
    approach.pose.position.z += approach_height_;
    approach.header = target.header;
    publishPose(approach, "抬升至目标上方");
    waitForPosition(approach);

    target.header.stamp = ros::Time::now();
    publishPose(target, "下移至抓取点（纯 Z 方向）");
    waitForPosition(target);
    // [修复B6] 原来这里空等 5 秒；实测停止下发位姿后腕部 5 秒内漂 57°。
    ros::Duration(1.0).sleep();

    double dist = 0.0, angle = 0.0;
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        dist = calcDistance(current_pose_, target);
        tf2::Quaternion qc(current_pose_.pose.orientation.x,
                           current_pose_.pose.orientation.y,
                           current_pose_.pose.orientation.z,
                           current_pose_.pose.orientation.w);
        tf2::Quaternion qt(target.pose.orientation.x,
                           target.pose.orientation.y,
                           target.pose.orientation.z,
                           target.pose.orientation.w);
        angle = qc.angleShortestPath(qt);
    }
    // [修复B6] 实测：到达抓取点后停止下发位姿，仅 5 秒腕部姿态就从 0.02° 漂到
    // 57.44°，在这个错位姿态上合爪等于夹空气 —— 这才是"根本夹不起来"的直接原因。
    // 对策：① 不再空等 5 秒；② 偏差大先重新收敛；③ 闭合全程持续下发目标位姿。
    // 另外把锁的作用域收窄到只读位姿，原版把锁扣到闭合结束，姿态回调被堵 6 秒。
    if (dist > pos_tolerance_ || angle > ori_tolerance_)
    {
        ROS_WARN("抓取点偏差 (%.3fm, %.2f°)，先重新收敛再闭合",
                 dist, angle * 180.0 / M_PI);
        waitForPosition(target);
    }
    controlGripperIncremental(true, "夹爪闭合");
    waitForGripper(target);
    grasped_ = gripper_closed_;
    grasp_value_ = gripper_value_;
    grasp_effort_ = live_effort_;

    init_pose_.header.stamp = ros::Time::now();
    publishPose(init_pose_, "回到初始位姿");
    waitForPosition(init_pose_);
    // [修复B7] 抬起后复核手指受力：掉下去就说明只是碰到了，没夹住。
    lift_effort_ = live_effort_;
    ROS_INFO("[修复B7] 抬起后手指受力 %.3f（仿真自带的接触门槛是 0.1）", lift_effort_);

    place_pose_.pose.position.x = (conveyor_side_ == "right" ? right_place_x_ : left_place_x_);
    place_pose_.header.stamp = ros::Time::now();
    publishPose(place_pose_, "移动至放置位姿1");
    waitForPosition(place_pose_);

    geometry_msgs::PoseStamped set_pose = place_pose_;
    set_pose.header.stamp = ros::Time::now();
    set_pose.pose.position.z -= 0.2;
    publishPose(set_pose, "移动至放置位姿2");
    waitForPosition(set_pose);
    ros::Duration(3.0).sleep();

    openGripperDirectly("夹爪打开");
    ros::Duration(2.0).sleep();

    geometry_msgs::PoseStamped get_pose = set_pose;
    get_pose.header.stamp = ros::Time::now();
    get_pose.pose.position.z += 0.2;
    publishPose(get_pose, "移动至放置位姿3");
    waitForPosition(get_pose);
    ros::Duration(3.0).sleep();

    init_pose_.header.stamp = ros::Time::now();
    publishPose(init_pose_, "回到初始位姿");
    waitForPosition(init_pose_);

    std_msgs::Bool res_msg;
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        double dist = calcDistance(current_pose_, init_pose_);
        tf2::Quaternion qc(current_pose_.pose.orientation.x,
                           current_pose_.pose.orientation.y,
                           current_pose_.pose.orientation.z,
                           current_pose_.pose.orientation.w);
        tf2::Quaternion qt(init_pose_.pose.orientation.x,
                           init_pose_.pose.orientation.y,
                           init_pose_.pose.orientation.z,
                           init_pose_.pose.orientation.w);
        double angle = qc.angleShortestPath(qt);
        (void)dist;
        (void)angle;
        // [修复B3] 原版"成功"只表示回到了初始位姿 —— 实测假阳性。
        // 现在要求：夹爪确实合拢到有夹持力的行程，且收到过接触信号。
        // [修复B7] 不再用接触信号判成功 —— 本仿真里它只表示"夹爪本体撞到东西"。
        // 改用仿真自带的接触门槛（jaka_env.py: effort_threshold = 0.1）：
        // 必须①手指真的合到底 ②抬起后受力仍达标（说明物体跟着走）。
        pick_success_ = (grasp_value_ >= 0.03f) && (lift_effort_ >= 0.1f);
    }
    res_msg.data = pick_success_;
    result_pub_.publish(res_msg);
    ROS_INFO("本次抓放流程 %s（闭合值 %.4f，合爪受力 %.3f，抬起后受力 %.3f，接触信号 %s）",
             pick_success_ ? "成功" : "失败", grasp_value_, grasp_effort_, lift_effort_,
             grasped_ ? "有" : "无");

    {
        std::lock_guard<std::mutex> lock(mutex_);
        busy_ = false;
    }
    ROS_INFO("抓放任务完成");
}

int main(int argc, char **argv)
{
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "pick_and_place_node");
    ros::NodeHandle nh;
    PickAndPlaceNode node(nh);
    ros::spin();
    return 0;
}
