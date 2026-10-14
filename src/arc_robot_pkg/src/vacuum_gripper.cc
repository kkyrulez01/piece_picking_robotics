#include "arc_robot_pkg/vacuum_gripper.hh"

#include <gz/sim/components/Name.hh>
#include <gz/sim/components/DetachableJoint.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <iostream>

using namespace custom_plugins;

// Constructor initializes variables
VacuumGripper::VacuumGripper():
    suction_active(false),
    currently_holding(false),
    model_entity(gz::sim::kNullEntity),
    suction_cup_entity(gz::sim::kNullEntity),
    fixed_joint_entity(gz::sim::kNullEntity)
{
}

// Destructor
VacuumGripper::~VacuumGripper() = default;

// Callback logic (OnSwitch)
void VacuumGripper::OnSwitch(const gz::msgs::Boolean &_msg) {
    suction_active = _msg.data();

    std::cout << "[VacuumGripper] OnSwitch received: "
          << suction_active << std::endl;
}

// Configure Logic
// Configure() runs once Gazebo loads the plugin
void VacuumGripper::Configure(
    const gz::sim::Entity &_entity,
    const std::shared_ptr<const sdf::Element> &_sdf,
    gz::sim::EntityComponentManager &_ecm,
    gz::sim::EventManager &_eventMgr)
{   
    (void)_eventMgr;
    model_entity = _entity;

    // Reads plugin settings from SDF
    suction_link_name = _sdf->Get<std::string>("link_name");
    std::string topic = _sdf->Get<std::string>("topic", "/gripper/switch").first;
    
    // Find suction link
    gz::sim::Model model(_entity);
    suction_cup_entity = model.LinkByName(_ecm, suction_link_name);
    std::cout << "[VacuumGripper] Plugin loaded" << std::endl;

    // Subscribes to switch topic
    node.Subscribe(topic, &VacuumGripper::OnSwitch, this);
}

// PreUpdate logic
void VacuumGripper::PreUpdate(
    const gz::sim::UpdateInfo &_info,
    gz::sim::EntityComponentManager &_ecm)
{
    if (_info.paused) return;

    // Retry finding the suction gripper link if not found during Configure()
    if (suction_cup_entity == gz::sim::kNullEntity && model_entity != gz::sim::kNullEntity) {
        
        gz::sim::Model model(model_entity);
        suction_cup_entity = model.LinkByName(
            _ecm,
            suction_link_name
        );

        if (suction_cup_entity != gz::sim::kNullEntity) {
            std::cout << "[VaccumGripper] Found suction_cup_entity later: "
                    << suction_cup_entity << std::endl;
        }
    }

    // Picking object
    if (suction_active && !currently_holding) {
        gz::sim::Entity target_object = _ecm.EntityByComponents(
            gz::sim::components::Name("block_link")
        );

        std::cout << "[VacuumGripper] Trying attach. "
          << "suction_cup_entity=" << suction_cup_entity
          << ", target_object=" << target_object
          << std::endl;

        // Don't create joint if the block hasn't spawned yet
        if (target_object != gz::sim::kNullEntity && suction_cup_entity != gz::sim::kNullEntity){
            fixed_joint_entity = _ecm.CreateEntity();
            
            // Dynamically create a joint so that the gripper and target object is "welded" together
            _ecm.CreateComponent(
                fixed_joint_entity,
                gz::sim::components::DetachableJoint({suction_cup_entity, target_object, "fixed"})
            );
            currently_holding = true;
        }
    }

    // Placing object
    else if (!suction_active && currently_holding) {
        _ecm.RequestRemoveEntity(fixed_joint_entity); // Delete the newly created joint
        fixed_joint_entity = gz::sim::kNullEntity;
        currently_holding = false;
    }
}

// Register the plugin
GZ_ADD_PLUGIN(
    custom_plugins::VacuumGripper,
    gz::sim::System,
    custom_plugins::VacuumGripper::ISystemConfigure,
    custom_plugins::VacuumGripper::ISystemPreUpdate
)

GZ_ADD_PLUGIN_ALIAS(
    custom_plugins::VacuumGripper,
    "custom_plugins::VacuumGripper"
)