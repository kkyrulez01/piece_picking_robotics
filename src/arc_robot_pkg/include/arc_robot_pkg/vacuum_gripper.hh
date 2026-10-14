#ifndef ARC_ROBOT_PKG_VACUUM_GRIPPER_HH_
#define ARC_ROBOT_PKG_VACUUM_GRIPPER_HH_

#include <memory>
#include <string>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/boolean.pb.h>

namespace custom_plugins
{
    class VacuumGripper:
        public gz::sim::System,
        public gz::sim::ISystemConfigure,
        public gz::sim::ISystemPreUpdate
    {
        public:
            // Constructor and Destructor
            VacuumGripper();
            ~VacuumGripper() override;

            // Gazebo System overrides
            void Configure(const gz::sim::Entity &_entity, 
                const std::shared_ptr<const sdf::Element> &_sdf,
                gz::sim::EntityComponentManager &_ecm,
                gz::sim::EventManager &_eventMgr) override;
            
            void PreUpdate(const gz::sim::UpdateInfo &_info,
                gz::sim::EntityComponentManager &_ecm) override;
            
        private:
            // Topic Callback
            void OnSwitch(const gz::msgs::Boolean &_msg);

            // Memory Variables
            gz::transport::Node node;
            bool suction_active;
            bool currently_holding;

            std::string suction_link_name;

            gz::sim::Entity model_entity;
            gz::sim::Entity suction_cup_entity;
            gz::sim::Entity fixed_joint_entity;
    };
}

#endif // ARC_ROBOT_PKG_VACUUM_GRIPPER_HH_