#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/EntityComponentManager.hh>

#include <gz/sim/components/ExternalWorldWrenchCmd.hh>

#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>

#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/wrench.pb.h>

#include <sdf/Element.hh>

#include <iostream>
#include <memory>
#include <mutex>
#include <string>

namespace gz::sim
{
class ConstantWrenchSystem
    : public System,
      public ISystemConfigure,
      public ISystemPreUpdate
{
public:
  void Configure(const Entity &_entity,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 EntityComponentManager &_ecm,
                 EventManager &) override
  {
    this->model = Model(_entity);
    if (!this->model.Valid(_ecm))
    {
      std::cerr << "[ConstantWrenchSystem] ERROR: invalid model entity\n";
      return;
    }

    if (!_sdf || !_sdf->HasElement("link_name"))
    {
      std::cerr << "[ConstantWrenchSystem] ERROR: <link_name> is required\n";
      return;
    }
    this->linkName = _sdf->Get<std::string>("link_name");

    // Topics (optional overrides)
    if (_sdf->HasElement("topic"))
      this->wrenchTopic = _sdf->Get<std::string>("topic");
    else
      this->wrenchTopic = "/model/" + this->model.Name(_ecm) + "/external_wrench_cmd";

    if (_sdf->HasElement("enable_topic"))
      this->enableTopic = _sdf->Get<std::string>("enable_topic");
    else
      this->enableTopic = "/model/" + this->model.Name(_ecm) + "/external_wrench_enable";

    if (_sdf->HasElement("enabled"))
      this->enabled = _sdf->Get<bool>("enabled");

    if (_sdf->HasElement("default_force"))
    {
      auto v = _sdf->Get<gz::math::Vector3d>("default_force");
      this->forceX = v.X(); this->forceY = v.Y(); this->forceZ = v.Z();
    }
    if (_sdf->HasElement("default_torque"))
    {
      auto v = _sdf->Get<gz::math::Vector3d>("default_torque");
      this->torqueX = v.X(); this->torqueY = v.Y(); this->torqueZ = v.Z();
    }

    // Find target link
    this->linkEntity = this->model.LinkByName(_ecm, this->linkName);
    if (this->linkEntity == kNullEntity)
    {
      std::cerr << "[ConstantWrenchSystem] ERROR: link '" << this->linkName
                << "' not found in model '" << this->model.Name(_ecm) << "'.\n";
      return;
    }

    // Subscribe
    this->node.Subscribe(this->wrenchTopic,
      &ConstantWrenchSystem::OnWrenchMsg, this);
    this->node.Subscribe(this->enableTopic,
      &ConstantWrenchSystem::OnEnableMsg, this);

    std::cout << "[ConstantWrenchSystem] Ready\n"
              << "  model:        " << this->model.Name(_ecm) << "\n"
              << "  link:         " << this->linkName << "\n"
              << "  wrench topic: " << this->wrenchTopic << "\n"
              << "  enable topic: " << this->enableTopic << "\n"
              << "  enabled:      " << (this->enabled ? "true" : "false") << "\n"
              << "  frame:        WORLD (ExternalWorldWrenchCmd)\n";
  }

  void PreUpdate(const UpdateInfo &,
                 EntityComponentManager &_ecm) override
  {
    if (this->linkEntity == kNullEntity)
      return;

    double fx, fy, fz, tx, ty, tz;
    bool enabledLocal;
    {
      std::lock_guard<std::mutex> lock(this->mtx);
      enabledLocal = this->enabled;
      fx = this->forceX; fy = this->forceY; fz = this->forceZ;
      tx = this->torqueX; ty = this->torqueY; tz = this->torqueZ;
    }

    if (!enabledLocal)
      return;

    gz::msgs::Wrench wrenchMsg;
    auto *f = wrenchMsg.mutable_force();
    f->set_x(fx); f->set_y(fy); f->set_z(fz);
    auto *tau = wrenchMsg.mutable_torque();
    tau->set_x(tx); tau->set_y(ty); tau->set_z(tz);

    auto comp = _ecm.Component<components::ExternalWorldWrenchCmd>(this->linkEntity);
    if (!comp)
    {
      _ecm.CreateComponent(this->linkEntity,
                           components::ExternalWorldWrenchCmd(wrenchMsg));
    }
    else
    {
      comp->Data() = wrenchMsg;
      _ecm.SetChanged(this->linkEntity,
                      components::ExternalWorldWrenchCmd::typeId,
                      ComponentState::OneTimeChange);
    }
  }

private:
  void OnWrenchMsg(const gz::msgs::Wrench &_msg)
  {
    std::lock_guard<std::mutex> lock(this->mtx);
    this->forceX = _msg.force().x();
    this->forceY = _msg.force().y();
    this->forceZ = _msg.force().z();
    this->torqueX = _msg.torque().x();
    this->torqueY = _msg.torque().y();
    this->torqueZ = _msg.torque().z();
  }

  void OnEnableMsg(const gz::msgs::Boolean &_msg)
  {
    std::lock_guard<std::mutex> lock(this->mtx);
    this->enabled = _msg.data();
  }

private:
  Model model{kNullEntity};
  Entity linkEntity{kNullEntity};
  std::string linkName;

  gz::transport::Node node;
  std::string wrenchTopic;
  std::string enableTopic;

  std::mutex mtx;
  bool enabled{false};

  double forceX{0.0}, forceY{0.0}, forceZ{0.0};
  double torqueX{0.0}, torqueY{0.0}, torqueZ{0.0};
};

GZ_ADD_PLUGIN(ConstantWrenchSystem,
              System,
              ConstantWrenchSystem::ISystemConfigure,
              ConstantWrenchSystem::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(ConstantWrenchSystem, "gz::sim::ConstantWrenchSystem")
} // namespace gz::sim